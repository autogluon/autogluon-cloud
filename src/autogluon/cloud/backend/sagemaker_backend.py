import copy
import json
import logging
import os
import tarfile
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import sagemaker
import yaml
from botocore.exceptions import ClientError

from autogluon.common.loaders import load_pd, load_pkl
from autogluon.common.utils.s3_utils import is_s3_url, s3_path_to_bucket_prefix

from ..data import FormatConverterFactory
from ..endpoint.sagemaker_endpoint import SagemakerEndpoint
from ..job import SageMakerBatchTransformationJob, SageMakerFitJob
from ..scripts import ScriptManager
from ..utils.ag_sagemaker import (
    AutoGluonBatchPredictor,
    AutoGluonNonRepackInferenceModel,
    AutoGluonRealtimePredictor,
    AutoGluonRepackInferenceModel,
)
from ..utils.aws_utils import setup_sagemaker_session
from ..utils.constants import SAGEMAKER_RESOURCE_PREFIX, VALID_ACCEPT
from ..utils.iam import (
    IAM_POLICY_FILE_NAME,
    SAGEMAKER_CLOUD_POLICY,
    SAGEMAKER_TRUST_RELATIONSHIP,
    TRUST_RELATIONSHIP_FILE_NAME,
    replace_iam_policy_place_holder,
    replace_trust_relationship_place_holder,
)
from ..utils.sagemaker_utils import parse_framework_version
from ..utils.utils import (
    convert_image_path_to_encoded_bytes_in_dataframe,
    is_image_file,
    split_pred_and_pred_proba,
    unzip_file,
    zipfolder,
)
from .backend import Backend
from .constant import SAGEMAKER

logger = logging.getLogger(__name__)


class SagemakerBackend(Backend):
    def __init__(self, local_output_path: str, cloud_output_path: str, predictor_type: str, **kwargs) -> None:
        self.initialize(
            local_output_path=local_output_path,
            cloud_output_path=cloud_output_path,
            predictor_type=predictor_type,
            **kwargs,
        )

    @property
    def name(self) -> str:
        """Name of this backend"""
        return SAGEMAKER

    def initialize(self, local_output_path: str, cloud_output_path: str, predictor_type: str, **kwargs) -> None:
        """Initialize the backend."""
        try:
            self.role_arn = sagemaker.get_execution_role()
        except ClientError as e:
            logger.warning(
                "Failed to get IAM role. Did you configure and authenticate the IAM role?",
                "For more information, https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-role.html",
                f"If you do not have a role created yet, \
                You can use {self.__class__.__name__}.generate_trust_relationship_and_iam_policy_file() to get the required trust relationship and iam policy",
                "You can then use the generated trust relationship and IAM policy to create an IAM role",
                "For more information, https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create.html",
                "IMPORTANT: Please review the generated trust relationship and IAM policy before you create an IAM role with them",
            )
            raise e
        self.local_output_path = local_output_path
        self.cloud_output_path = cloud_output_path
        self.predictor_type = predictor_type
        self.sagemaker_session = setup_sagemaker_session()
        self.endpoint = None
        self._region = self.sagemaker_session.boto_region_name
        self._fit_job: SageMakerFitJob = SageMakerFitJob(session=self.sagemaker_session)
        self._realtime_predictor_cls = AutoGluonRealtimePredictor

    def generate_default_permission(
        self, account_id: str, cloud_output_bucket: str, output_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate required trust relationship and IAM policy file in json format for CloudPredictor with SageMaker backend.
        Users can use the generated files to create an IAM role for themselves.
        IMPORTANT: Make sure you review both files before creating the role!

        Parameters
        ----------
        account_id: str
            The AWS account ID you plan to use for CloudPredictor.
        cloud_output_bucket: str
            s3 bucket name where intermediate artifacts will be uploaded and trained models should be saved.
            You need to create this bucket beforehand and we would put this bucket in the policy being created.
        output_path: str
            Where you would like the generated file being written to.
            If not specified, will write to the current folder.

        Return
        ------
        A dict containing the trust relationship and IAM policy files paths
        """
        if output_path is None:
            output_path = "."
        trust_relationship_file_path = os.path.join(output_path, TRUST_RELATIONSHIP_FILE_NAME)
        iam_policy_file_path = os.path.join(output_path, IAM_POLICY_FILE_NAME)

        trust_relationship = replace_trust_relationship_place_holder(
            trust_relationship_document=SAGEMAKER_TRUST_RELATIONSHIP, account_id=account_id
        )
        iam_policy = replace_iam_policy_place_holder(
            policy_document=SAGEMAKER_CLOUD_POLICY, account_id=account_id, bucket=cloud_output_bucket
        )
        with open(trust_relationship_file_path, "w") as file:
            json.dump(trust_relationship, file, indent=4)

        with open(iam_policy_file_path, "w") as file:
            json.dump(iam_policy, file, indent=4)

        logger.info(f"Generated trust relationship to {trust_relationship_file_path}")
        logger.info(f"Generated iam policy to {iam_policy_file_path}")
        logger.info(
            "IMPORTANT: Please review the trust relationship and iam policy before you use them to create an IAM role"
        )
        logger.info(
            "Please refer to AWS documentation on how to create an IAM role: https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create.html"
        )

        return {"trust_relationship": trust_relationship_file_path, "iam_policy": iam_policy_file_path}

    def parse_backend_fit_kwargs(self, kwargs: Dict) -> List[Dict]:
        """Parse backend specific kwargs and get them ready to be sent to fit call"""
        autogluon_sagemaker_estimator_kwargs = kwargs.get("autogluon_sagemaker_estimator_kwargs", None)
        fit_kwargs = kwargs.get("fit_kwargs", None)

        return [autogluon_sagemaker_estimator_kwargs, fit_kwargs]

    def attach_job(self, job_name: str) -> None:
        """
        Attach to a existing training job.
        This is useful when the local process crashed and you want to reattach to the previous job

        Parameters
        ----------
        job_name: str
            The name of the job being attached
        """
        self._fit_job = SageMakerFitJob.attach(job_name)

    def get_fit_job_status(self) -> str:
        """
        Get the status of the training job.
        This is useful when the user made an asynchronous call to the `fit()` function

        Returns
        -------
        str,
            Status of the job
        """
        return self._fit_job.get_job_status()

    def fit(
        self,
        *,
        predictor_init_args: Dict[str, Any],
        predictor_fit_args: Dict[str, Any],
        image_column: Optional[str] = None,
        leaderboard: bool = True,
        framework_version: str = "latest",
        job_name: Optional[str] = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        volume_size: int = 100,
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        autogluon_sagemaker_estimator_kwargs: Dict = None,
        **kwargs,
    ) -> None:
        """
        Fit the predictor with SageMaker.
        This function will first upload necessary config and train data to s3 bucket.
        Then launch a SageMaker training job with the AutoGluon training container.

        Parameters
        ----------
        predictor_init_args: dict
            Init args for the predictor
        predictor_fit_args: dict
            Fit args for the predictor
        image_column: str, default = None
            The column name in the training/tuning data that contains the image paths.
            The image paths MUST be absolute paths to you local system.
        leaderboard: bool, default = True
            Whether to include the leaderboard in the output artifact
        framework_version: str, default = `latest`
            Training container version of autogluon.
            If `latest`, will use the latest available container version.
            If provided a specific version, will use this version.
            If `custom_image_uri` is set, this argument will be ignored.
        job_name: str, default = None
            Name of the launched training job.
            If None, CloudPredictor will create one with prefix ag-cloudpredictor
        instance_type: str, default = 'ml.m5.2xlarge'
            Instance type the predictor will be trained on with SageMaker.
        instance_count: int, default = 1
            Number of instance used to fit the predictor.
        volumes_size: int, default = 30
            Size in GB of the EBS volume to use for storing input data during training (default: 30).
            Must be large enough to store training data if File Mode is used (which is the default).
        wait: bool, default = True
            Whether the call should wait until the job completes
            To be noticed, the function won't return immediately because there are some preparations needed prior fit.
            Use `get_fit_job_status` to get job status.
        autogluon_sagemaker_estimator_kwargs: dict, default = dict()
            Any extra arguments needed to initialize AutoGluonSagemakerEstimator
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Framework for all options
        **kwargs:
            Any extra arguments needed to pass to fit.
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Framework.fit for all options
        """
        predictor_fit_args = copy.deepcopy(predictor_fit_args)
        train_data = predictor_fit_args.pop("train_data")
        tune_data = predictor_fit_args.pop("tuning_data", None)
        if custom_image_uri:
            framework_version, py_version = None, None
            logger.log(20, f"Training with custom_image_uri=={custom_image_uri}")
        else:
            framework_version, py_version = parse_framework_version(
                framework_version, "training", minimum_version="0.6.0"
            )
            logger.log(20, f"Training with framework_version=={framework_version}")

        if not job_name:
            job_name = sagemaker.utils.unique_name_from_base(SAGEMAKER_RESOURCE_PREFIX)

        if autogluon_sagemaker_estimator_kwargs is None:
            autogluon_sagemaker_estimator_kwargs = {}
        autogluon_sagemaker_estimator_kwargs = copy.deepcopy(autogluon_sagemaker_estimator_kwargs)
        autogluon_sagemaker_estimator_kwargs.pop("output_path", None)
        if (
            autogluon_sagemaker_estimator_kwargs.get("disable_profiler", None) is None
            and autogluon_sagemaker_estimator_kwargs.get("debugger_hook_config", None) is None
        ):
            autogluon_sagemaker_estimator_kwargs["disable_profiler"] = True
            autogluon_sagemaker_estimator_kwargs["debugger_hook_config"] = False
        output_path = self.cloud_output_path + "/model"
        code_location = self.cloud_output_path + "/code"

        self._train_script_path = ScriptManager.get_train_script(self.predictor_type, framework_version)
        entry_point = self._train_script_path
        user_entry_point = autogluon_sagemaker_estimator_kwargs.pop("entry_point", None)
        if user_entry_point:
            logger.warning(
                f"Providing a custom entry point could break the fit. Please refer to `{entry_point}` for our implementation"
            )
            entry_point = user_entry_point
        else:
            # Avoid user passing in source_dir without specifying entry point
            autogluon_sagemaker_estimator_kwargs.pop("source_dir", None)

        config_args = dict(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            leaderboard=leaderboard,
        )
        if image_column is not None:
            config_args["image_column"] = image_column
        config = self._construct_config(**config_args)
        inputs = self._upload_fit_artifact(
            train_data=train_data,
            tune_data=tune_data,
            config=config,
            image_column=image_column,
            serving_script=ScriptManager.get_serve_script(
                self.predictor_type, framework_version
            ),  # Training and Inference should have the same framework_version
        )

        self._fit_job.run(
            role=self.role_arn,
            entry_point=entry_point,
            region=self._region,
            instance_type=instance_type,
            instance_count=instance_count,
            volume_size=volume_size,
            framework_version=framework_version,
            py_version=py_version,
            base_job_name="autogluon-cloudpredictor-train",
            output_path=output_path,
            code_location=code_location,
            inputs=inputs,
            custom_image_uri=custom_image_uri,
            wait=wait,
            job_name=job_name,
            autogluon_sagemaker_estimator_kwargs=autogluon_sagemaker_estimator_kwargs,
            **kwargs,
        )

    def parse_backend_deploy_kwargs(self, kwargs: Dict) -> List[Dict]:
        """Parse backend specific kwargs and get them ready to be sent to deploy call"""
        model_kwargs = kwargs.get("model_kwargs", None)
        deploy_kwargs = kwargs.get("deploy_kwargs", None)

        return [model_kwargs, deploy_kwargs]

    def prepare_deploy(self, realtime_predictor_cls, **kwargs) -> None:
        """Things to be configured before deploy goes here"""
        self._realtime_predictor_cls = realtime_predictor_cls

    def deploy(
        self,
        predictor_path: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        framework_version: str = "latest",
        instance_type: str = "ml.m5.2xlarge",
        initial_instance_count: int = 1,
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        model_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """
        Deploy a predictor as a SageMaker endpoint, which can be used to do real-time inference later.
        This method would first create a AutoGluonSagemakerInferenceModel with the trained predictor,
        and then deploy it to the endpoint.

        Parameters
        ----------
        predictor_path: str
            Path to the predictor tarball you want to deploy.
            Path can be both a local path or a S3 location.
            If None, will deploy the most recent trained predictor trained with `fit()`.
        endpoint_name: str
            The endpoint name to use for the deployment.
            If None, CloudPredictor will create one with prefix `ag-cloudpredictor`
        framework_version: str, default = `latest`
            Inference container version of autogluon.
            If `latest`, will use the latest available container version.
            If provided a specific version, will use this version.
            If `custom_image_uri` is set, this argument will be ignored.
        instance_type: str, default = 'ml.m5.2xlarge'
            Instance to be deployed for the endpoint
        initial_instance_count: int, default = 1,
            Initial number of instances to be deployed for the endpoint
        wait: Bool, default = True,
            Whether to wait for the endpoint to be deployed.
            To be noticed, the function won't return immediately because there are some preparations needed prior deployment.
        model_kwargs: dict, default = dict()
            Any extra arguments needed to initialize Sagemaker Model
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#model for all options
        **kwargs:
            Any extra arguments needed to pass to deploy.
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#sagemaker.model.Model.deploy for all options
        """
        assert (
            self.endpoint is None
        ), "There is an endpoint already attached. Either detach it with `detach` or clean it up with `cleanup_deployment`"
        if not predictor_path:
            predictor_path = self._fit_job.get_output_path()
            assert predictor_path, "No cloud trained model found."
        predictor_path = self._upload_predictor(predictor_path, f"endpoints/{endpoint_name}/predictor")

        if not endpoint_name:
            endpoint_name = sagemaker.utils.unique_name_from_base(SAGEMAKER_RESOURCE_PREFIX)
        if custom_image_uri:
            framework_version, py_version = None, None
            logger.log(20, f"Deploying with custom_image_uri=={custom_image_uri}")
        else:
            framework_version, py_version = parse_framework_version(
                framework_version, "inference", minimum_version="0.6.0"
            )
            logger.log(20, f"Deploying with framework_version=={framework_version}")

        self._serve_script_path = ScriptManager.get_serve_script(self.predictor_type, framework_version)
        entry_point = self._serve_script_path
        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs = copy.deepcopy(model_kwargs)
        user_entry_point = model_kwargs.pop("entry_point", None)
        if user_entry_point:
            logger.warning(
                f"Providing a custom entry point could break the deployment. Please refer to `{entry_point}` for our implementation"
            )
            entry_point = user_entry_point

        repack_model = False
        if predictor_path != self._fit_job.get_output_path() or user_entry_point is not None:
            # Not inference on cloud trained model or not using inference on cloud trained model
            # Need to repack the code into model. This will slow down batch inference and deployment
            repack_model = True
        predictor_cls = self._realtime_predictor_cls
        user_predictor_cls = model_kwargs.pop("predictor_cls", None)
        if user_predictor_cls:
            logger.warning(
                "Providing a custom predictor_cls could break the deployment.",
                "Please refer to `AutoGluonRealtimePredictor` for how to provide a custom predictor",
            )
            predictor_cls = user_predictor_cls

        if repack_model:
            model_cls = AutoGluonRepackInferenceModel
        else:
            model_cls = AutoGluonNonRepackInferenceModel
        model = model_cls(
            model_data=predictor_path,
            role=self.role_arn,
            region=self._region,
            framework_version=framework_version,
            py_version=py_version,
            instance_type=instance_type,
            custom_image_uri=custom_image_uri,
            entry_point=entry_point,
            predictor_cls=predictor_cls,
            **model_kwargs,
        )

        logger.log(20, "Deploying model to the endpoint")
        self.endpoint = SagemakerEndpoint(
            model.deploy(
                endpoint_name=endpoint_name,
                instance_type=instance_type,
                initial_instance_count=initial_instance_count,
                wait=wait,
                **kwargs,
            )
        )

    def attach_endpoint(self, endpoint: Union[str, SagemakerEndpoint]) -> None:
        """
        Attach the current backend to an existing SageMaker endpoint.

        Parameters
        ----------
        endpoint: str or  :class:`SagemakerEndpoint`
            If str is passed, it should be the name of the endpoint being attached to.
        """
        assert (
            self.endpoint is None
        ), "There is an endpoint already attached. Either detach it with `detach` or clean it up with `cleanup_deployment`"
        if type(endpoint) == str:
            endpoint = self._realtime_predictor_cls(
                endpoint_name=endpoint,
                sagemaker_session=self.sagemaker_session,
            )
            self.endpoint = SagemakerEndpoint(endpoint)
        elif isinstance(endpoint, SagemakerEndpoint):
            self.endpoint = endpoint
        else:
            raise ValueError(f"Please provide either an endpoint name or an endpoint of type `{SagemakerEndpoint}`")

    def detach_endpoint(self) -> SagemakerEndpoint:
        """Detach the current endpoint and return it"""
        assert self.endpoint is not None, "There is no attached endpoint"
        detached_endpoint = self.endpoint
        self.endpoint = None
        return detached_endpoint

    def predict_realtime(self, test_data: Union[str, pd.DataFrame], **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """Realtime prediction with the endpoint"""
        raise NotImplementedError

    def predict(self, test_data: Union[str, pd.DataFrame], **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """Batch inference"""
        raise NotImplementedError

    def _construct_config(self, predictor_init_args, predictor_fit_args, leaderboard, **kwargs):
        assert self.predictor_type is not None
        config = dict(
            predictor_type=self.predictor_type,
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            leaderboard=leaderboard,
            **kwargs,
        )
        path = os.path.join(self.local_output_path, "utils", "config.yaml")
        with open(path, "w") as f:
            yaml.dump(config, f)
        return path

    def _prepare_data(self, data, filename, output_type="csv"):
        path = os.path.join(self.local_output_path, "utils")
        converter = FormatConverterFactory.get_converter(output_type)
        return converter.convert(data, path, filename)

    def _find_common_path_and_replace_image_column(self, data, image_column):
        common_path = os.path.commonpath(data[image_column].tolist())
        common_path_head = os.path.split(common_path)[0]  # we keep the base dir to match zipping behavior
        data[image_column] = data[image_column].apply(lambda path: os.path.relpath(path, common_path_head))

        return data, common_path

    def _upload_fit_artifact(
        self,
        train_data,
        tune_data,
        config,
        serving_script,
        image_column=None,
    ):
        cloud_bucket, cloud_key_prefix = s3_path_to_bucket_prefix(self.cloud_output_path)
        util_key_prefix = cloud_key_prefix + "/utils"

        common_train_data_path = None
        common_tune_data_path = None
        if image_column is not None:
            # Find common path to zip and replace image column with relative path to be used in remote environment
            if isinstance(train_data, str):
                train_data = load_pd.load(train_data)
            else:
                train_data = copy.deepcopy(train_data)
            if tune_data is not None:
                if isinstance(tune_data, str):
                    tune_data = load_pd.load(tune_data)
                else:
                    tune_data = copy.deepcopy(tune_data)
            train_data, common_train_data_path = self._find_common_path_and_replace_image_column(
                data=train_data, image_column=image_column
            )
            if tune_data is not None:
                tune_data, common_tune_data_path = self._find_common_path_and_replace_image_column(
                    data=tune_data, image_column=image_column
                )

        train_input = train_data
        train_data = self._prepare_data(train_data, "train")
        logger.log(20, "Uploading train data...")
        train_input = self.sagemaker_session.upload_data(
            path=train_data, bucket=cloud_bucket, key_prefix=util_key_prefix
        )
        logger.log(20, "Train data uploaded successfully")

        tune_input = tune_data
        if tune_data is not None:
            tune_data = self._prepare_data(tune_data, "tune")
            logger.log(20, "Uploading tune data...")
            tune_input = self.sagemaker_session.upload_data(
                path=tune_data, bucket=cloud_bucket, key_prefix=util_key_prefix
            )
            logger.log(20, "Tune data uploaded successfully")

        config_input = self.sagemaker_session.upload_data(path=config, bucket=cloud_bucket, key_prefix=util_key_prefix)

        serving_input = self.sagemaker_session.upload_data(
            path=serving_script, bucket=cloud_bucket, key_prefix=util_key_prefix
        )

        train_images_input = self._upload_fit_image_artifact(
            image_dir_path=common_train_data_path, bucket=cloud_bucket, key_prefix=util_key_prefix
        )
        tune_images_input = self._upload_fit_image_artifact(
            image_dir_path=common_tune_data_path, bucket=cloud_bucket, key_prefix=util_key_prefix
        )
        inputs = dict(train=train_input, config=config_input, serving=serving_input)
        if tune_input is not None:
            inputs["tune"] = tune_input
        if train_images_input is not None:
            inputs["train_images"] = train_images_input
        if tune_images_input is not None:
            inputs["tune_images"] = tune_images_input

        return inputs

    def _upload_fit_image_artifact(self, image_dir_path, bucket, key_prefix):
        upload_image_path = None
        if image_dir_path is not None:
            image_zip_filename = image_dir_path
            assert os.path.isdir(image_dir_path), "Please provide a folder containing the images"
            image_zip_filename = os.path.basename(os.path.normpath(image_dir_path))
            logger.log(20, "Zipping images ...")
            zipfolder(image_zip_filename, image_dir_path)
            image_zip_filename += ".zip"
            logger.log(20, "Uploading images ...")
            upload_image_path = self.sagemaker_session.upload_data(
                path=image_zip_filename,
                bucket=bucket,
                key_prefix=key_prefix,
            )
            logger.log(20, "Images uploaded successfully")
        return upload_image_path

    def _upload_predictor(self, predictor_path, key_prefix):
        cloud_bucket, _ = s3_path_to_bucket_prefix(self.cloud_output_path)
        if not is_s3_url(predictor_path):
            if os.path.isfile(predictor_path):
                if tarfile.is_tarfile(predictor_path):
                    predictor_path = self.sagemaker_session.upload_data(
                        path=predictor_path, bucket=cloud_bucket, key_prefix=key_prefix
                    )
                else:
                    raise ValueError("Please provide a tarball containing the model")
            else:
                raise ValueError("Please provide a valid path to the model tarball.")
        return predictor_path
