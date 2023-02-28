import copy
import json
import logging
import os
import tarfile
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import sagemaker
import yaml
from botocore.exceptions import ClientError
from sagemaker import Predictor

from autogluon.common.loaders import load_pd
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
from ..utils.misc import MostRecentInsertedOrderedDict
from ..utils.sagemaker_utils import parse_framework_version
from ..utils.utils import (
    convert_image_path_to_encoded_bytes_in_dataframe,
    is_image_file,
    split_pred_and_pred_proba,
    zipfolder,
)
from .backend import Backend
from .constant import SAGEMAKER

logger = logging.getLogger(__name__)


class SagemakerBackend(Backend):
    name = SAGEMAKER

    def __init__(self, local_output_path: str, cloud_output_path: str, predictor_type: str, **kwargs) -> None:
        self.initialize(
            local_output_path=local_output_path,
            cloud_output_path=cloud_output_path,
            predictor_type=predictor_type,
            **kwargs,
        )

    @property
    def _realtime_predictor_cls(self) -> Predictor:
        """Class used for realtime endpoint"""
        return AutoGluonRealtimePredictor

    def initialize(self, local_output_path: str, cloud_output_path: str, predictor_type: str, **kwargs) -> None:
        """Initialize the backend."""
        try:
            self.role_arn = sagemaker.get_execution_role()
        except ClientError as e:
            logger.warning(
                "Failed to get IAM role. Did you configure and authenticate the IAM role?",
                "For more information, https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-role.html",
                f"If you do not have a role created yet, \
                You can use {self.__class__.__name__}.generate_default_permission() to get the required trust relationship and iam policy",
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
        self._batch_transform_jobs = MostRecentInsertedOrderedDict()

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

    def parse_backend_fit_kwargs(self, kwargs: Dict) -> Dict[str, Any]:
        """Parse backend specific kwargs and get them ready to be sent to fit call"""
        autogluon_sagemaker_estimator_kwargs = kwargs.get("autogluon_sagemaker_estimator_kwargs", None)
        fit_kwargs = kwargs.get("fit_kwargs", None)

        return dict(autogluon_sagemaker_estimator_kwargs=autogluon_sagemaker_estimator_kwargs, fit_kwargs=fit_kwargs)

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

    @property
    def is_fit(self) -> bool:
        """Whether the backend is fitted"""
        return self._fit_job.completed

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

    def get_fit_job_output_path(self) -> str:
        """
        Get the output path in the cloud of the trained artifact

        Returns
        -------
        str,
            Output path of the job
        """
        return self._fit_job.get_output_path()

    def get_fit_job_info(self) -> Dict[str, Any]:
        """
        Get general info of the training job.

        Returns
        -------
        Dict,
            General info of the job
        """
        return self._fit_job.info()

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
        autogluon_sagemaker_estimator_kwargs: Optional[Dict] = None,
        fit_kwargs: Optional[Dict] = None,
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
        volume_size: int, default = 100
            Size in GB of the EBS volume to use for storing input data during training (default: 100).
            Must be large enough to store training data if File Mode is used (which is the default).
        wait: bool, default = True
            Whether the call should wait until the job completes
            To be noticed, the function won't return immediately because there are some preparations needed prior fit.
            Use `get_fit_job_status` to get job status.
        autogluon_sagemaker_estimator_kwargs: dict, default = dict()
            Any extra arguments needed to initialize AutoGluonSagemakerEstimator
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Framework for all options
        fit_kwargs:
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
        if fit_kwargs is None:
            fit_kwargs = {}

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
            **fit_kwargs,
        )

    def parse_backend_deploy_kwargs(self, kwargs: Dict) -> Dict[str, Any]:
        """Parse backend specific kwargs and get them ready to be sent to deploy call"""
        model_kwargs = kwargs.get("model_kwargs", None)
        deploy_kwargs = kwargs.get("deploy_kwargs", None)

        return dict(model_kwargs=model_kwargs, deploy_kwargs=deploy_kwargs)

    def deploy(
        self,
        predictor_path: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        framework_version: str = "latest",
        instance_type: str = "ml.m5.2xlarge",
        initial_instance_count: int = 1,
        custom_image_uri: Optional[str] = None,
        volume_size: int = 100,
        wait: bool = True,
        model_kwargs: Optional[Dict] = None,
        deploy_kwargs: Optional[Dict] = None,
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
        volume_size: int, default = 100
           The size, in GB, of the ML storage volume attached to individual inference instance associated with the production variant.
           Currenly only Amazon EBS gp2 storage volumes are supported.
        wait: Bool, default = True,
            Whether to wait for the endpoint to be deployed.
            To be noticed, the function won't return immediately because there are some preparations needed prior deployment.
        model_kwargs: dict, default = dict()
            Any extra arguments needed to initialize Sagemaker Model
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#model for all options
        deploy_kwargs:
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
        model_kwargs_env = model_kwargs.pop("env", None)
        SAGEMAKER_MODEL_SERVER_WORKERS = "SAGEMAKER_MODEL_SERVER_WORKERS"
        if model_kwargs_env is not None:
            if (
                SAGEMAKER_MODEL_SERVER_WORKERS in model_kwargs_env
                and int(model_kwargs_env[SAGEMAKER_MODEL_SERVER_WORKERS]) > 1
            ):
                logger.warning(
                    f"Setting {SAGEMAKER_MODEL_SERVER_WORKERS} to value larger than 1 might cause running out of RAM and/or GPU RAM"
                )
            else:
                model_kwargs_env[SAGEMAKER_MODEL_SERVER_WORKERS] = "1"
        else:
            model_kwargs_env = {SAGEMAKER_MODEL_SERVER_WORKERS: "1"}

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
            env=model_kwargs_env,
            **model_kwargs,
        )
        if deploy_kwargs is None:
            deploy_kwargs = {}

        logger.log(20, "Deploying model to the endpoint")
        self.endpoint = SagemakerEndpoint(
            model.deploy(
                endpoint_name=endpoint_name,
                instance_type=instance_type,
                initial_instance_count=initial_instance_count,
                volume_size=volume_size,
                wait=wait,
                **deploy_kwargs,
            )
        )

    def cleanup_deployment(self) -> None:
        """
        Delete endpoint, endpoint configuration and deployed model
        """
        assert self.endpoint is not None, "No deployed endpoint detected"
        self.endpoint.delete_endpoint()
        self.endpoint = None

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

    def predict_real_time(
        self,
        test_data: Union[str, pd.DataFrame],
        test_data_image_column: Optional[str] = None,
        accept: str = "application/x-parquet",
        **kwargs,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Predict with the deployed SageMaker endpoint. A deployed SageMaker endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict()` instead.

        Parameters
        ----------
        test_data: Union(str, pandas.DataFrame)
            The test data to be inferenced. Can be a pandas.DataFrame, or a local path to csv file.
        test_data_image_column: default = None
            If test_data involves image modality, you must specify the column name corresponding to image paths.
            The path MUST be an abspath
        accept: str, default = application/x-parquet
            Type of accept output content.
            Valid options are application/x-parquet, text/csv, application/json

        Returns
        -------
        Pandas.Series
        Predict results in Series
        """
        self._validate_predict_real_time_args(accept)
        test_data = self._load_predict_real_time_test_data(test_data, test_data_image_column=test_data_image_column)
        pred, _ = self._predict_real_time(test_data=test_data, accept=accept)

        return pred

    def predict_proba_real_time(
        self,
        test_data: Union[str, pd.DataFrame],
        test_data_image_column: Optional[str] = None,
        accept: str = "application/x-parquet",
        **kwargs,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Predict probability with the deployed SageMaker endpoint. A deployed SageMaker endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict_proba()` instead.
        If your problem_type is regression, this functions identically to `predict_real_time`, returning the same output.

        Parameters
        ----------
        test_data: Union(str, pandas.DataFrame)
            The test data to be inferenced. Can be a pandas.DataFrame, or a local path to csv file.
        test_data_image_column: default = None
            If test_data involves image modality, you must specify the column name corresponding to image paths.
            The path MUST be an abspath
        accept: str, default = application/x-parquet
            Type of accept output content.
            Valid options are application/x-parquet, text/csv, application/json

        Returns
        -------
        Pandas.DataFrame or Pandas.Series
            Will return a Pandas.Series when it's a regression problem. Will return a Pandas.DataFrame otherwise
        """
        self._validate_predict_real_time_args(accept)
        test_data = self._load_predict_real_time_test_data(test_data, test_data_image_column=test_data_image_column)
        pred, proba = self._predict_real_time(test_data=test_data, accept=accept)

        if proba is None:
            return pred

        return proba

    def parse_backend_predict_kwargs(self, kwargs: Dict) -> Dict[str, Any]:
        """Parse backend specific kwargs and get them ready to be sent to predict call"""
        download = kwargs.get("download", True)
        persist = kwargs.get("persist", True)
        save_path = kwargs.get("persist", None)
        model_kwargs = kwargs.get("model_kwargs", None)
        transformer_kwargs = kwargs.get("transformer_kwargs", None)
        transform_kwargs = kwargs.get("transform_kwargs", None)

        return dict(
            download=download,
            persist=persist,
            save_path=save_path,
            model_kwargs=model_kwargs,
            transformer_kwargs=transformer_kwargs,
            transform_kwargs=transform_kwargs,
        )

    def get_batch_inference_job_info(self, job_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get general info of the batch inference job.
        If job_name not specified, return the info of the most recent batch inference job

        Returns
        -------
        Optional[Dict[str, Any]],
            A dictinary containing general info of the job.
        """
        if not job_name:
            job_name = self._batch_transform_jobs.last
        job: SageMakerBatchTransformationJob = self._batch_transform_jobs.get(job_name, None)
        if job:
            return job.info()
        return None

    def get_batch_inference_job_status(self, job_name: Optional[str] = None) -> str:
        """
        Get general status of the batch inference job.
        If job_name not specified, return the info of the most recent batch inference job

        Returns
        -------
        str,
        Valid Values: InProgress | Completed | Failed | Stopping | Stopped | NotCreated
        """
        if not job_name:
            job_name = self._batch_transform_jobs.last
        job: SageMakerBatchTransformationJob = self._batch_transform_jobs.get(job_name, None)
        if job:
            return job.get_job_status()
        return "NotCreated"

    def get_batch_inference_jobs(self) -> List[str]:
        """
        Get a list of names of all batch inference jobs

        Returns
        -------
        List[str],
            a list of names of all batch inference jobs
        """
        return [job_name for job_name in self._batch_transform_jobs.keys()]

    def predict(
        self,
        test_data: Union[str, pd.DataFrame],
        test_data_image_column: Optional[str] = None,
        predictor_path: Optional[str] = None,
        framework_version: str = "latest",
        job_name: Optional[str] = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        download: bool = True,
        persist: bool = True,
        save_path: Optional[str] = None,
        model_kwargs: Optional[Dict] = None,
        transformer_kwargs: Optional[Dict] = None,
        transform_kwargs: Optional[Dict] = None,
    ) -> Optional[pd.Series]:
        """
        Predict using SageMaker batch transform.
        When minimizing latency isn't a concern, then the batch transform functionality may be easier, more scalable, and more appropriate.
        If you want to minimize latency, use `predict_real_time()` instead.
        To learn more: https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html
        This method would first create a AutoGluonSagemakerInferenceModel with the trained predictor,
        then create a transformer with it, and call transform in the end.

        Parameters
        ----------
        test_data: Union(str, pandas.DataFrame)
            The test data to be inferenced. Can be a pandas.DataFrame, or a local path to a csv.
        test_data_image_column: str, default = None
            If test_data involves image modality, you must specify the column name corresponding to image paths.
            The path MUST be an abspath
        predictor_path: str
            Path to the predictor tarball you want to use to predict.
            Path can be both a local path or a S3 location.
            If None, will use the most recent trained predictor trained with `fit()`.
        framework_version: str, default = `latest`
            Inference container version of autogluon.
            If `latest`, will use the latest available container version.
            If provided a specific version, will use this version.
            If `custom_image_uri` is set, this argument will be ignored.
        job_name: str, default = None
            Name of the launched training job.
            If None, CloudPredictor will create one with prefix ag-cloudpredictor.
        instance_count: int, default = 1,
            Number of instances used to do batch transform.
        instance_type: str, default = 'ml.m5.2xlarge'
            Instance to be used for batch transform.
        wait: bool, default = True
            Whether to wait for batch transform to complete.
            To be noticed, the function won't return immediately because there are some preparations needed prior transform.
        download: bool, default = True
            Whether to download the batch transform results to the disk and load it after the batch transform finishes.
            Will be ignored if `wait` is `False`.
        persist: bool, default = True
            Whether to persist the downloaded batch transform results on the disk.
            Will be ignored if `download` is `False`
        save_path: str, default = None,
            Path to save the downloaded result.
            Will be ignored if `download` is `False`.
            If None, CloudPredictor will create one.
            If `persist` is `False`, file would first be downloaded to this path and then removed.
        model_kwargs: dict, default = dict()
            Any extra arguments needed to initialize Sagemaker Model
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#model for all options
        transformer_kwargs: dict
            Any extra arguments needed to pass to transformer.
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html#sagemaker.transformer.Transformer for all options.
        transform_kwargs:
            Any extra arguments needed to pass to transform.
            Please refer to
            https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html#sagemaker.transformer.Transformer.transform for all options.

        Returns
        -------
        Optional Pandas.Series
            Predict results in Series if `download` is True
            None if `download` is False
        """
        pred, _ = self._predict(
            test_data=test_data,
            test_data_image_column=test_data_image_column,
            predictor_path=predictor_path,
            framework_version=framework_version,
            job_name=job_name,
            instance_type=instance_type,
            instance_count=instance_count,
            custom_image_uri=custom_image_uri,
            wait=wait,
            download=download,
            persist=persist,
            save_path=save_path,
            model_kwargs=model_kwargs,
            transformer_kwargs=transformer_kwargs,
            transform_kwargs=transform_kwargs,
        )

        return pred

    def predict_proba(
        self,
        test_data: Union[str, pd.DataFrame],
        test_data_image_column: Optional[str] = None,
        include_predict: bool = True,
        predictor_path: Optional[str] = None,
        framework_version: str = "latest",
        job_name: Optional[str] = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        download: bool = True,
        persist: bool = True,
        save_path: Optional[str] = None,
        model_kwargs: Optional[Dict] = None,
        transformer_kwargs: Optional[Dict] = None,
        transform_kwargs: Optional[Dict] = None,
    ) -> Optional[Union[Tuple[pd.Series, Union[pd.DataFrame, pd.Series]], Union[pd.DataFrame, pd.Series]]]:
        """
        Predict using SageMaker batch transform.
        When minimizing latency isn't a concern, then the batch transform functionality may be easier, more scalable, and more appropriate.
        If you want to minimize latency, use `predict_real_time()` instead.
        To learn more: https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html
        This method would first create a AutoGluonSagemakerInferenceModel with the trained predictor,
        then create a transformer with it, and call transform in the end.

        Parameters
        ----------
        test_data: Union(str, pandas.DataFrame)
            The test data to be inferenced. Can be a pandas.DataFrame, or a local path to a csv.
        test_data_image_column: str, default = None
            If test_data involves image modality, you must specify the column name corresponding to image paths.
            The path MUST be an abspath
        include_predict: bool, default = True
            Whether to include predict result along with predict_proba results.
            This flag can save you time from making two calls to get both the prediction and the probability as batch inference involves noticeable overhead.
        predictor_path: str
            Path to the predictor tarball you want to use to predict.
            Path can be both a local path or a S3 location.
            If None, will use the most recent trained predictor trained with `fit()`.
        framework_version: str, default = `latest`
            Inference container version of autogluon.
            If `latest`, will use the latest available container version.
            If provided a specific version, will use this version.
            If `custom_image_uri` is set, this argument will be ignored.
        job_name: str, default = None
            Name of the launched training job.
            If None, CloudPredictor will create one with prefix ag-cloudpredictor.
        instance_count: int, default = 1,
            Number of instances used to do batch transform.
        instance_type: str, default = 'ml.m5.2xlarge'
            Instance to be used for batch transform.
        wait: bool, default = True
            Whether to wait for batch transform to complete.
            To be noticed, the function won't return immediately because there are some preparations needed prior transform.
        download: bool, default = True
            Whether to download the batch transform results to the disk and load it after the batch transform finishes.
            Will be ignored if `wait` is `False`.
        persist: bool, default = True
            Whether to persist the downloaded batch transform results on the disk.
            Will be ignored if `download` is `False`
        save_path: str, default = None,
            Path to save the downloaded result.
            Will be ignored if `download` is `False`.
            If None, CloudPredictor will create one.
            If `persist` is `False`, file would first be downloaded to this path and then removed.
        model_kwargs: dict, default = dict()
            Any extra arguments needed to initialize Sagemaker Model
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#model for all options
        transformer_kwargs: dict
            Any extra arguments needed to pass to transformer.
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html#sagemaker.transformer.Transformer for all options.
        transform_kwargs:
            Any extra arguments needed to pass to transform.
            Please refer to
            https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html#sagemaker.transformer.Transformer.transform for all options.

        Returns
        -------
        Optional[Union[Tuple[pd.Series, Union[pd.DataFrame, pd.Series]], Union[pd.DataFrame, pd.Series]]]
            If `download` is False, will return None or (None, None) if `include_predict` is True
            If `download` is True and `include_predict` is True,
            will return (prediction, predict_probability), where prediction is a Pandas.Series and predict_probability is a Pandas.DataFrame
            or a Pandas.Series that's identical to prediction when it's a regression problem.
        """
        pred, pred_proba = self._predict(
            test_data=test_data,
            test_data_image_column=test_data_image_column,
            predictor_path=predictor_path,
            framework_version=framework_version,
            job_name=job_name,
            instance_type=instance_type,
            instance_count=instance_count,
            custom_image_uri=custom_image_uri,
            wait=wait,
            download=download,
            persist=persist,
            save_path=save_path,
            model_kwargs=model_kwargs,
            transformer_kwargs=transformer_kwargs,
            transform_kwargs=transform_kwargs,
        )

        if include_predict:
            return pred, pred_proba

        return pred_proba

    def download_predict_results(self, job_name: Optional[str] = None, save_path: Optional[str] = None) -> str:
        """
        Download batch transform result

        Parameters
        ----------
        job_name: str
            The specific batch transform job results to download.
            If None, will download the most recent job results.
        save_path: str
            Path to save the downloaded results.
            If None, CloudPredictor will create one.

        Returns
        -------
        str,
            Path to downloaded results.
        """
        if not job_name:
            job_name = self._batch_transform_jobs.last
        assert job_name is not None, "There is no batch transform job."
        job = self._batch_transform_jobs.get(job_name, None)
        assert job is not None, f"Could not find the batch transform job that matches name {job_name}"
        result_path = job.get_output_path()
        assert result_path is not None, "No predict results found."
        file_name = result_path.split("/")[-1]
        if not save_path:
            save_path = self.local_output_path
        save_path = os.path.expanduser(save_path)
        save_path = os.path.abspath(save_path)
        results_save_path = os.path.join(save_path, "batch_transform", job_name)
        if not os.path.isdir(results_save_path):
            os.makedirs(results_save_path)
        results_bucket, results_key_prefix = s3_path_to_bucket_prefix(result_path)
        self.sagemaker_session.download_data(
            path=results_save_path, bucket=results_bucket, key_prefix=results_key_prefix
        )
        results_save_path = os.path.join(results_save_path, file_name)
        logger.log(20, f"Batch results have been downloaded to {results_save_path}")

        return results_save_path

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

    def _validate_predict_real_time_args(self, accept):
        assert self.endpoint is not None, "Please call `deploy()` to deploy an endpoint first."
        assert accept in VALID_ACCEPT, f"Invalid accept type: {accept}. Options are {VALID_ACCEPT}."

    def _load_predict_real_time_test_data(self, test_data, test_data_image_column):
        if isinstance(test_data, str):
            test_data = load_pd.load(test_data)
        if isinstance(test_data, pd.DataFrame):
            if test_data_image_column is not None:
                test_data = convert_image_path_to_encoded_bytes_in_dataframe(test_data, test_data_image_column)

        return test_data

    def _predict_real_time(self, test_data, accept, split_pred_proba=True, **initial_args):
        try:
            prediction = self.endpoint.predict(test_data, initial_args={"Accept": accept, **initial_args})
            pred, pred_proba = None, None
            pred = prediction
            if split_pred_proba:
                pred, pred_proba = split_pred_and_pred_proba(prediction)
            return pred, pred_proba
        except ClientError as e:
            if e.response["Error"]["Code"] == "413":  # Error code for pay load too large
                logger.warning(
                    "The invocation of endpoint failed with Error Code 413. This is likely due to pay load size being too large."
                )
                logger.warning(
                    "SageMaker endpoint could only take maximum 5MB. Please consider reduce test data size or use `predict()` instead."
                )
            raise e

    def _upload_batch_predict_data(self, test_data, bucket, key_prefix):
        # If a directory of images, upload directly
        if isinstance(test_data, str) and not os.path.isdir(test_data):
            # either a file to a dataframe, or a file to an image
            if is_image_file(test_data):
                logger.warning(
                    "Are you sure you want to do batch inference on a single image? You might want to try `deploy()` and `predict_real_time()` instead"
                )
            else:
                test_data = load_pd.load(test_data)

        if isinstance(test_data, pd.DataFrame):
            test_data = self._prepare_data(test_data, "test", output_type="csv")
        logger.log(20, "Uploading data...")
        test_input = self.sagemaker_session.upload_data(path=test_data, bucket=bucket, key_prefix=key_prefix + "/data")
        logger.log(20, "Data uploaded successfully")

        return test_input

    def _predict(
        self,
        test_data,
        test_data_image_column=None,
        predictor_path=None,
        framework_version="latest",
        job_name=None,
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        custom_image_uri=None,
        wait=True,
        download=True,
        persist=True,
        save_path=None,
        model_kwargs=None,
        transformer_kwargs=None,
        split_pred_proba=True,
        transform_kwargs=None,
    ):
        if not predictor_path:
            predictor_path = self._fit_job.get_output_path()
            assert predictor_path, "No cloud trained model found."

        if custom_image_uri:
            framework_version, py_version = None, None
            logger.log(20, f"Predicting with custom_image_uri=={custom_image_uri}")
        else:
            framework_version, py_version = parse_framework_version(
                framework_version, "inference", minimum_version="0.6.0"
            )
            logger.log(20, f"Predicting with framework_version=={framework_version}")

        if transform_kwargs is None:
            transform_kwargs = {}
        output_path = transform_kwargs.get("output_path", None)
        if not output_path:
            output_path = self.cloud_output_path
        assert is_s3_url(output_path)
        output_path = output_path + "/batch_transform" + f"/{sagemaker.utils.sagemaker_timestamp()}"

        cloud_bucket, cloud_key_prefix = s3_path_to_bucket_prefix(output_path)
        logger.log(20, "Preparing autogluon predictor...")
        predictor_path = self._upload_predictor(predictor_path, cloud_key_prefix + "/predictor")

        if not job_name:
            job_name = sagemaker.utils.unique_name_from_base(SAGEMAKER_RESOURCE_PREFIX)

        if test_data_image_column is not None:
            logger.warning("Batch inference with image modality could be slow because of some technical details.")
            logger.warning(
                "You can always retrieve the model trained with CloudPredictor and do batch inference using your custom solution."
            )
            if isinstance(test_data, str):
                test_data = load_pd.load(test_data)
            test_data = convert_image_path_to_encoded_bytes_in_dataframe(
                dataframe=test_data, image_column=test_data_image_column
            )
        test_input = self._upload_batch_predict_data(test_data, cloud_bucket, cloud_key_prefix)

        self._serve_script_path = ScriptManager.get_serve_script(self.predictor_type, framework_version)
        entry_point = self._serve_script_path
        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs = copy.deepcopy(model_kwargs)
        if transformer_kwargs is None:
            transformer_kwargs = {}
        transformer_kwargs = copy.deepcopy(transformer_kwargs)
        user_entry_point = model_kwargs.pop("entry_point", None)
        repack_model = False
        if predictor_path != self._fit_job.get_output_path() or user_entry_point is not None:
            # Not inference on cloud trained model or not using inference on cloud trained model
            # Need to repack the code into model. This will slow down batch inference and deployment
            repack_model = True
        if user_entry_point:
            entry_point = user_entry_point

        predictor_cls = AutoGluonBatchPredictor
        user_predictor_cls = model_kwargs.pop("predictor_cls", None)
        if user_predictor_cls:
            logger.warning(
                "Providing a custom predictor_cls could break the deployment. Please refer to `AutoGluonBatchPredictor` for how to provide a custom predictor"
            )
            predictor_cls = user_predictor_cls

        transform_kwargs = copy.deepcopy(transform_kwargs)
        content_type = transform_kwargs.pop("content_type", None)
        if "split_type" not in transform_kwargs:
            split_type = "Line"
        else:
            split_type = transform_kwargs.pop("split_type")
        if not content_type:
            content_type = "text/csv"

        if not wait:
            if download:
                logger.warning(
                    f"`download={download}` will be ignored because `wait={wait}`. Setting `download` to `False`."
                )
                download = False
        if not download:
            if persist:
                logger.warning(
                    f"`persist={persist}` will be ignored because `download={download}`. Setting `persist` to `False`."
                )
                persist = False
            if save_path:
                logger.warning(
                    f"`save_path={save_path}` will be ignored because `download={download}`. Setting `save_path` to `None`."
                )
                save_path = None

        batch_transform_job = SageMakerBatchTransformationJob(session=self.sagemaker_session)
        batch_transform_job.run(
            model_data=predictor_path,
            role=self.role_arn,
            region=self._region,
            framework_version=framework_version,
            py_version=py_version,
            instance_count=instance_count,
            instance_type=instance_type,
            entry_point=entry_point,
            predictor_cls=predictor_cls,
            output_path=output_path + "/results",
            test_input=test_input,
            job_name=job_name,
            split_type=split_type,
            content_type=content_type,
            custom_image_uri=custom_image_uri,
            wait=wait,
            transformer_kwargs=transformer_kwargs,
            model_kwargs=model_kwargs,
            repack_model=repack_model,
            **transform_kwargs,
        )
        self._batch_transform_jobs[job_name] = batch_transform_job

        pred, pred_proba = None, None
        if download:
            results_path = self.download_predict_results(save_path=save_path)
            # Batch inference will only return json format
            results = pd.read_json(results_path)
            pred = results
            if split_pred_proba:
                pred, pred_proba = split_pred_and_pred_proba(results)
        if not persist:
            os.remove(results_path)

        return pred, pred_proba

    def __getstate__(self) -> Dict[str, Any]:
        """Custom implementation of the pickle process"""
        d = self.__dict__.copy()
        d["sagemaker_session"] = None
        d["_region"] = None
        if self.endpoint is not None:
            d["_endpoint_saved"] = self.endpoint.endpoint_name
            d["endpoint"] = None

        return d

    def __setstate__(self, state):
        """Custom implementation of the unpickle process"""
        self.__dict__.update(state)
        self.sagemaker_session = setup_sagemaker_session()
        self._region = self.sagemaker_session.boto_region_name
        if hasattr(self, "_endpoint_saved") and self._endpoint_saved is not None:
            self.endpoiont = self.attach_endpoint(self._endpoint_saved)
            self._endpoint_saved = None
        self._fit_job.session = self.sagemaker_session
        for job in self._batch_transform_jobs:
            job.session = self.sagemaker_session
