from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import pandas as pd
import copy
import logging

from ..endpoint.endpoint import Endpoint
from .backend import Backend
from ..job.ray_job import RayFitJob


logger = logging.getLogger(__name__)


class RayBackend(Backend):
    name = "ray_backend"

    def initialize(self, local_output_path: str, cloud_output_path: str, predictor_type: str, **kwargs) -> None:
        """Initialize the backend."""
        self.local_output_path = local_output_path
        self.cloud_output_path = cloud_output_path
        self.predictor_type = predictor_type
        self._fit_job = RayFitJob(output_path=cloud_output_path + "/model")

    def generate_default_permission(self, **kwargs) -> Dict[str, str]:
        """Generate default permission file user could use to setup the corresponding entity, i.e. IAM Role in AWS"""
        raise NotImplementedError

    def parse_backend_fit_kwargs(self, kwargs: Dict) -> Dict[str, Any]:
        """Parse backend specific kwargs and get them ready to be sent to fit call"""
        raise NotImplementedError

    def attach_job(self, job_name: str) -> None:
        """
        Attach to an existing training job.
        This is useful when the local process crashed and you want to reattach to the previous job

        Parameters
        ----------
        job_name: str
            The name of the job being attached
        """
        raise NotImplementedError

    @property
    def is_fit(self) -> bool:
        """Whether the backend is fitted"""
        return self._fit_job.completed

    def get_fit_job_status(self) -> str:
        """
        Get the status of the training job.
        This is useful when the user made an asynchronous call to the `fit()` function
        """
        raise NotImplementedError

    def get_fit_job_output_path(self) -> str:
        """Get the output path in the cloud of the trained artifact"""
        raise NotImplementedError

    def get_fit_job_info(self) -> Dict[str, Any]:
        """
        Get general info of the training job.
        """
        raise NotImplementedError

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
        volume_size: int = 256,
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
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
        """
        if image_column is not None:
            logger.warning("Distributed training doesn't support image modality yet. Will ignore")
            image_column = None
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

    def parse_backend_deploy_kwargs(self, kwargs: Dict) -> Dict[str, Any]:
        """Parse backend specific kwargs and get them ready to be sent to deploy call"""
        raise NotImplementedError

    def deploy(self, **kwargs) -> None:
        """Deploy and endpoint"""
        raise NotImplementedError

    def cleanup_deployment(self, **kwargs) -> None:
        """Delete endpoint, and cleanup other artifacts"""
        raise NotImplementedError

    def attach_endpoint(self, endpoint: Endpoint) -> None:
        """Attach the backend to an existing endpoint"""
        raise NotImplementedError

    def detach_endpoint(self) -> Endpoint:
        """Detach the current endpoint and return it"""
        raise NotImplementedError

    def predict_real_time(self, test_data: Union[str, pd.DataFrame], **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """Realtime prediction with the endpoint"""
        raise NotImplementedError

    def predict_proba_real_time(self, test_data: Union[str, pd.DataFrame], **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """Realtime prediction probability with the endpoint"""
        raise NotImplementedError

    def parse_backend_predict_kwargs(self, kwargs: Dict) -> Dict[str, Any]:
        """Parse backend specific kwargs and get them ready to be sent to predict call"""
        raise NotImplementedError

    def get_batch_inference_job_info(self, job_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get general info of the batch inference job.
        If job_name not specified, return the info of the most recent batch inference job
        """
        raise NotImplementedError

    def get_batch_inference_job_status(self, job_name: Optional[str] = None) -> str:
        """
        Get general status of the batch inference job.
        If job_name not specified, return the info of the most recent batch inference job
        """
        raise NotImplementedError

    def get_batch_inference_jobs(self) -> List[str]:
        """Get a list of names of all batch inference jobs"""
        raise NotImplementedError

    def predict(self, test_data: Union[str, pd.DataFrame], **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """Batch inference"""
        raise NotImplementedError

    def predict_proba(self, test_data: Union[str, pd.DataFrame], **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """Batch inference probability"""
        raise NotImplementedError
