from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sagemaker import image_uris

from ..cluster.ray_cluster_config_generator import RayClusterConfigGenerator
from ..cluster.ray_cluster_manager import RayClusterManager
from ..data import FormatConverterFactory
from ..endpoint.endpoint import Endpoint
from ..job.ray_job import RayFitJob
from ..scripts import ScriptManager
from ..utils.constants import CLOUD_RESOURCE_PREFIX
from ..utils.dlc_utils import parse_framework_version
from ..utils.ray_aws_iam import RAY_INSTANCE_PROFILE_NAME
from ..utils.s3_utils import s3_path_to_bucket_prefix, upload_file
from ..utils.utils import get_utc_timestamp_now
from .backend import Backend
from .constant import RAY

logger = logging.getLogger(__name__)


class RayBackend(Backend):
    name = RAY

    @property
    def _cluster_config_generator() -> RayClusterConfigGenerator:
        return RayClusterConfigGenerator

    @property
    def _cluster_manager() -> RayClusterManager:
        return RayClusterManager

    @property
    def _config_file_name() -> str:
        return "ag_ray_cluster_config.yaml"

    def initialize(self, local_output_path: str, cloud_output_path: str, predictor_type: str, **kwargs) -> None:
        """Initialize the backend."""
        self.local_output_path = local_output_path
        self.cloud_output_path = cloud_output_path
        self.predictor_type = predictor_type
        self._fit_job = RayFitJob(output_path=cloud_output_path + "/model")

    def generate_default_permission(self, **kwargs) -> Dict[str, str]:
        """Generate default permission file user could use to setup the corresponding entity, i.e. IAM Role in AWS"""
        return RayClusterManager.generate_default_permission(**kwargs)

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
        self._fit_job = RayFitJob.attach(job_name)

    @property
    def is_fit(self) -> bool:
        """Whether the backend is fitted"""
        return self._fit_job.completed

    def get_fit_job_status(self) -> str:
        """
        Get the status of the training job.
        This is useful when the user made an asynchronous call to the `fit()` function
        """
        return self._fit_job.get_job_status()

    def get_fit_job_output_path(self) -> str:
        """Get the output path in the cloud of the trained artifact"""
        return self._fit_job.get_output_path()

    def get_fit_job_info(self) -> Dict[str, Any]:
        """
        Get general info of the training job.
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
        instance_count: Union[int, str] = "auto",
        volume_size: int = 256,
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        custom_config: Optional[Union[str, Dict[str, Any]]] = None,
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
        instance_count: Union[int, str], default = "auto",
            Number of instance used to fit the predictor.
            if not specified, will use isntance_count = number of folds to be trained to maximize parallalism
        volume_size: int, default = 256
            Size in GB of the EBS volume to use for storing input data during training (default: 256).
        wait: bool, default = True
            Whether the call should wait until the job completes
            To be noticed, the function won't return immediately because there are some preparations needed prior fit.
            Use `get_fit_job_status` to get job status.
        custom_config: Optional[Union[str, Dict[str, Any]]], default = None
            Config to be used to launch up the cluster. Default: None
            If not set, will use the default config pre-defined.
            If str, must be a path pointing to a yaml file containing the config.
        """
        if image_column is not None:
            logger.warning("Distributed training doesn't support image modality yet. Will ignore")
            image_column = None
        predictor_fit_args = copy.deepcopy(predictor_fit_args)
        train_data = predictor_fit_args.pop("train_data")
        tune_data = predictor_fit_args.pop("tuning_data", None)
        num_bag_folds = predictor_fit_args.get("num_bag_folds", 8)

        if instance_count == "auto":
            instance_count = num_bag_folds
        image_uri = self._get_image_uri(
            framework_version=framework_version, custom_image_uri=custom_image_uri, instance_type=instance_type
        )
        train_data, tune_data = self._upload_data(train_data=train_data, tune_data=tune_data)

        config = self._generate_config(
            config=custom_config,
            instance_type=instance_type,
            instance_count=instance_count,
            volumes_size=volume_size,
            custom_image_uri=image_uri,
        )
        cluster_manager = self._cluster_manager(config=config, cloud_output_bucket=self.cloud_output_path)
        cluster_up = False
        try:
            cluster_manager.up()
            cluster_up = True
            cluster_manager.setup_connection()
            if not job_name:
                job_name = CLOUD_RESOURCE_PREFIX + "-" + get_utc_timestamp_now()
            job = RayFitJob(output_path=self.cloud_output_path + "/model")
            train_script = ScriptManager.get_train_script(backend_type=self.name, framework_version=framework_version)
            predictor_init_args = json.dumps(predictor_init_args)
            predictor_fit_args = json.dumps(predictor_fit_args)
            entry_point_command = f"python3 {train_script} --predictor_init_args {predictor_init_args} --predictor_fit_args {predictor_fit_args} --train_data {train_data}"
            if tune_data is not None:
                entry_point_command += f" --tune_data {tune_data}"
            if leaderboard:
                entry_point_command += f" --leaderboard"
            job.run(entry_point=entry_point_command, job_name=job_name, wait=wait)
        except Exception as e:
            logger.warning("Exception occured. Will tear down the cluster if needed.")
            raise e
        finally:
            if cluster_up:
                try:
                    cluster_manager.down()
                except Exception as down_e:
                    logger.warning(
                        "Failed to tear down the cluster. Please go to the console to terminate instanecs manually"
                    )
                    raise down_e

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

    def _get_image_uri(self, framework_version: str, custom_image_uri: str, instance_type: str):
        image_uri = custom_image_uri
        if custom_image_uri:
            framework_version, py_version = None, None
            logger.log(20, f"Training with custom_image_uri=={custom_image_uri}")
        else:
            framework_version, py_version = parse_framework_version(
                framework_version, "training", minimum_version="0.7.0"
            )
            logger.log(20, f"Training with framework_version=={framework_version}")
            image_uri = image_uris(
                "autogluon",
                region=self.region,
                version=framework_version,
                py_version=py_version,
                image_scope="training",
                instance_type=instance_type,
            )
        return image_uri

    def _upload_data(
        self, train_data: Union[str, pd.DataFrame], tune_data: Optional[Union[str, pd.DataFrame]] = None
    ) -> Tuple[str, str]:
        cloud_bucket, cloud_key_prefix = s3_path_to_bucket_prefix(self.cloud_output_path)
        util_key_prefix = cloud_key_prefix + "/utils"
        train_data = self._prepare_data(train_data, "train")
        logger.log(20, "Uploading train data..")
        upload_file(file_name=train_data, bucket=cloud_bucket, prefix=util_key_prefix)
        logger.log(20, "Train data uploaded successfully")
        if tune_data is not None:
            logger.log(20, "Uploading tune data...")
            tune_data = self._prepare_data(tune_data, "tune")
            upload_file(file_name=tune_data, bucket=cloud_bucket, prefix=util_key_prefix)
            logger.log(20, "Tune data uploaded successfully")
        return train_data, tune_data

    def _prepare_data(self, data: Union[str, pd.DataFrame], filename: str, output_type: str = "csv"):
        path = os.path.join(self.local_output_path, "utils")
        converter = FormatConverterFactory.get_converter(output_type)
        return converter.convert(data, path, filename)

    def _generate_config(
        self,
        config: Optional[Union[str, Dict[str, Any]]] = None,
        instance_type: Optional[str] = None,
        instance_count: Optional[int] = None,
        volumes_size: Optional[int] = None,
        custom_image_uri: Optional[str] = None,
    ):
        config_generator = self._cluster_config_generator(config=config, region=self.region)
        if config is None:
            config_generator.update_config(
                instance_type=instance_type,
                instance_count=instance_count,
                volumes_size=volumes_size,
                custom_image_uri=custom_image_uri,
                head_instance_profile=RAY_INSTANCE_PROFILE_NAME,
                worker_instance_profile=RAY_INSTANCE_PROFILE_NAME,
            )
            config = os.path.join(self.local_output_path, "utils", self._config_file_name)
            config_generator.save_config(save_path=config)
        return config
