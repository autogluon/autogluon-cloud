import copy
import json
import logging
import os
from typing import Any, Dict, Optional, Union

import pandas as pd
import sagemaker
from botocore.exceptions import ClientError

from autogluon.common.utils.s3_utils import is_s3_url, s3_path_to_bucket_prefix

from ..backend.backend import Backend
from ..endpoint.endpoint import Endpoint
from ..job import SageMakerBatchTransformationJob, SageMakerFitJob
from ..scripts import ScriptManager
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

logger = logging.getLogger(__name__)


class SagemakerBackend(Backend):
    def __init__(self, **kwargs) -> None:
        self.initialize(**kwargs)

    def initialize(self, **kwargs) -> None:
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
        self.sagemaker_session = setup_sagemaker_session()
        self.endpoint = None
        self._region = self.sagemaker_session.boto_region_name
        self._fit_job = SageMakerFitJob(session=self.sagemaker_session)

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

        Returns
        -------
        `CloudPredictor` object. Returns self.
        """
        # TODO: move this assert to predictor
        assert (
            not self._fit_job.completed
        ), "Predictor is already fit! To fit additional models, create a new `CloudPredictor`"
        predictor_fit_args = copy.deepcopy(predictor_fit_args)
        train_data = predictor_fit_args.pop("train_data")
        tune_data = predictor_fit_args.pop("tuning_data", None)
        if custom_image_uri:
            framework_version, py_version = None, None
            logger.log(20, f"Training with custom_image_uri=={custom_image_uri}")
        else:
            framework_version, py_version = self._parse_framework_version(framework_version, "training")
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
        cloud_bucket, _ = s3_path_to_bucket_prefix(self.cloud_output_path)

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

        self._setup_bucket(cloud_bucket)
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
        return self

    @abstractmethod
    def deploy(self, **kwargs) -> Endpoint:
        """Deploy and endpoint"""
        raise NotImplementedError

    @abstractmethod
    def predict_realtime(self, test_data: Union[str, pd.DataFrame], **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """Realtime prediction with the endpoint"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, test_data: Union[str, pd.DataFrame], **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """Batch inference"""
        raise NotImplementedError
