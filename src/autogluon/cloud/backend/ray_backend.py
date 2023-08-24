from __future__ import annotations

import copy
import logging
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sagemaker import image_uris

from autogluon.common.utils.s3_utils import s3_bucket_prefix_to_path, s3_path_to_bucket_prefix

from ..cluster.ray_cluster_config_generator import RayClusterConfigGenerator
from ..cluster.ray_cluster_manager import RayClusterManager
from ..data import FormatConverterFactory
from ..endpoint.endpoint import Endpoint
from ..job.ray_job import RayFitJob
from ..scripts import ScriptManager
from ..utils.constants import CLOUD_RESOURCE_PREFIX
from ..utils.dlc_utils import parse_framework_version
from ..utils.ec2 import get_latest_ami
from ..utils.iam import get_instance_profile_arn
from ..utils.ray_aws_iam import RAY_INSTANCE_PROFILE_NAME
from ..utils.s3_utils import upload_file
from ..utils.utils import get_utc_timestamp_now
from .backend import Backend
from .constant import RAY

logger = logging.getLogger(__name__)


class RayBackend(Backend):
    name = RAY

    @property
    def _cluster_config_generator(self) -> RayClusterConfigGenerator:
        return RayClusterConfigGenerator

    @property
    def _cluster_manager(self) -> RayClusterManager:
        return RayClusterManager

    @property
    def _config_file_name(self) -> str:
        return "ag_ray_cluster_config.yaml"

    def initialize(self, **kwargs) -> None:
        """Initialize the backend."""
        super().initialize(**kwargs)
        self.region = None
        self._fit_job = RayFitJob()
        os.makedirs(os.path.join(self.local_output_path, "job"), exist_ok=True)

    @staticmethod
    def generate_default_permission(**kwargs) -> Dict[str, str]:
        """Generate default permission file user could use to setup the corresponding entity, i.e. IAM Role in AWS"""
        return RayClusterManager.generate_default_permission(**kwargs)

    def parse_backend_fit_kwargs(self, kwargs: Dict) -> Dict[str, Any]:
        """Parse backend specific kwargs and get them ready to be sent to fit call"""
        custom_config = kwargs.get("custom_config", None)
        cluster_name = kwargs.get("cluster_name", None)
        initialization_commands = kwargs.get("initialization_commands", None)
        ephemeral_cluster = kwargs.get("ephemeral_cluster", True)
        return dict(
            custom_config=custom_config,
            cluster_name=cluster_name,
            initialization_commands=initialization_commands,
            ephemeral_cluster=ephemeral_cluster,
        )

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
        instance_type: str = "ml.m5.2xlarge",  # SageMaker instance type needed to fetch image uri
        instance_count: Union[int, str] = "auto",
        volume_size: int = 256,
        custom_image_uri: Optional[str] = None,
        timeout: int = 24 * 60 * 60,
        wait: bool = True,
        ephemeral_cluster: bool = True,
        custom_config: Optional[Union[str, Dict[str, Any]]] = None,
        cluster_name: Optional[str] = None,
        initialization_commands: Optional[List[str]] = None,
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
            If not specified, will use isntance_count = number of folds to be trained to maximize parallalism
        volume_size: int, default = 256
            Size in GB of the EBS volume to use for storing input data during training (default: 256).
        timeout: int, default = 24*60*60
            Timeout in seconds for training. This timeout doesn't include time for pre-processing or launching up the training job.
        wait: bool, default = True
            Whether the call should wait until the job completes
            To be noticed, the function won't return immediately because there are some preparations needed prior fit.
            Use `get_fit_job_status` to get job status.
        ephemeral_cluster: bool, default = True
            Whether to tear down the cluster once the job finished or not.
            If set to False, the user would need to shutdown the cluster manually.
        custom_config: Optional[Union[str, Dict[str, Any]]], default = None
            Config to be used to launch up the cluster. Default: None
            If not set, will use the default config pre-defined.
            If str, must be a path pointing to a yaml file containing the config.
        cluster_name: Optional[str] = None, default = None
            The name of the cluster being launched.
            If not specified, will be auto-generated with format f"ag_ray_aws_default_{timestamp}".
            If custom_config is provided, this option will not overwrite cluster name in your custom_config
        initialization_commands: Optional[List[str]], default = None
            The initialization commands of the ray cluster.
            If not specified, will contain a default ECR login command to be able to pull AG DLC image, i.e.
                - aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
            To learn more about initialization_commands,
                https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-configuration.html#initialization-commands
        """
        if image_column is not None:
            raise ValueError("Distributed training doesn't support image modality yet")
        predictor_fit_args = copy.deepcopy(predictor_fit_args)
        train_data = predictor_fit_args.pop("train_data")
        tune_data = predictor_fit_args.pop("tuning_data", None)
        presets = predictor_fit_args.pop("presets", [])
        num_bag_folds = predictor_fit_args.get("num_bag_folds", None)
        hyperparameter_tune_kwargs = predictor_fit_args.get("hyperparameter_tune_kwargs", None)

        if instance_count == "auto":
            instance_count = num_bag_folds
        else:
            if (
                int(instance_count) > 1
                and "best_quality" not in presets
                and "high_quality" not in presets
                and "good_quality" not in presets
                and num_bag_folds is None
                and hyperparameter_tune_kwargs is None
            ):
                logger.warning(
                    f"Tabular Predictor will be trained without bagging nor HPO hence not distributed, but you specified instance count > 1: {instance_count}."
                )
                logger.warning("Will deploy cluster with 1 instance only to save costs")
                instance_count = 1
        if instance_count is None:
            if "best_quality" in presets or "high_quality" in presets or "good_quality" in presets:
                instance_count = 8
            else:
                instance_count = 1

        image_uri = self._get_image_uri(
            framework_version=framework_version, custom_image_uri=custom_image_uri, instance_type=instance_type
        )
        ag_args_path = os.path.join(self.local_output_path, "job", "ag_args.pkl")
        self.prepare_args(
            path=ag_args_path, predictor_init_args=predictor_init_args, predictor_fit_args=predictor_fit_args
        )
        train_script = ScriptManager.get_train_script(backend_type=self.name, framework_version=framework_version)
        job_path = os.path.join(self.local_output_path, "job")
        shutil.copy(train_script, job_path)
        train_data, tune_data = self._upload_data(train_data=train_data, tune_data=tune_data)

        if instance_type.startswith("ml."):
            # Remove the ml. prefix from SageMaker instance type
            instance_type = ".".join(instance_type.split(".")[1:])

        self._setup_role_and_permission()
        key_name = None
        key_local_path = None
        if custom_config is None:
            key_name = f"ag_ray_cluster_{get_utc_timestamp_now()}"
            key_local_path = os.path.join(self.local_output_path, "utils")
            key_local_path = self._setup_key(key_name=key_name, local_path=key_local_path)

        ami = get_latest_ami()

        config = self._generate_config(
            config=custom_config,
            cluster_name=cluster_name,
            instance_type=instance_type,
            instance_count=instance_count,
            volumes_size=volume_size,
            ami=ami,
            custom_image_uri=image_uri,
            ssh_key_path=key_local_path,
            initialization_commands=initialization_commands,
        )
        cluster_manager = self._cluster_manager(config=config, cloud_output_bucket=self.cloud_output_path)
        cluster_up = False
        job_submitted = False
        try:
            logger.log(20, "Launching up ray cluster")
            cluster_manager.up()
            cluster_up = True
            logger.log(20, "Waiting for 60s to give the cluster some buffer time")
            time.sleep(60)
            cluster_manager.setup_connection()
            time.sleep(10)  # waiting for connection to setup
            if job_name is None:
                job_name = CLOUD_RESOURCE_PREFIX + "-" + get_utc_timestamp_now()
            job = RayFitJob(output_path=self.cloud_output_path + "/model")
            self._fit_job = job

            entry_point_command = f"python3 {os.path.basename(train_script)} --ag_args_path {os.path.basename(ag_args_path)} --train_data {train_data} --model_output_path {self.get_fit_job_output_path()} --ray_job_id {job_name}"  # noqa: E501
            if tune_data is not None:
                entry_point_command += f" --tune_data {tune_data}"
            if leaderboard:
                entry_point_command += " --leaderboard"
            if not wait and ephemeral_cluster:
                entry_point_command += f" --cluster_config_file {os.path.basename(config)}"
            job.run(
                entry_point=entry_point_command,
                runtime_env={
                    "working_dir": job_path,
                    "env_vars": {
                        "AG_DISTRIBUTED_MODE": "1",
                        "AG_MODEL_SYNC_PATH": f"{self.cloud_output_path}/model_sync/",
                        "AG_UTIL_PATH": f"{self.cloud_output_path}/utils/",
                        "AG_NUM_NODES": str(instance_count),
                        # TODO: update syncing logic in tabular https://github.com/ray-project/ray/pull/37142
                        "RAY_AIR_REENABLE_DEPRECATED_SYNC_TO_HEAD_NODE": "1",
                    },
                },
                job_name=job_name,
                timeout=timeout,
                wait=wait,
            )
            job_submitted = True
            if wait and job.get_job_status() != "SUCCEEDED":
                raise ValueError("Training job failed. Please check the log for reason.")
        except Exception as e:
            logger.warning("Exception occured. Will tear down the cluster if needed.")
            raise e
        finally:
            if wait:
                if ephemeral_cluster:
                    if cluster_up:
                        self._tear_down_cluster(
                            cluster_manager,
                            key_name=key_name,
                            key_local_path=os.path.join(self.local_output_path, "utils"),
                        )
                else:
                    logger.log(
                        20,
                        "Cluster not being destroyed because `ephemeral_cluster` set to False. Please destroy the cluster yourself.",
                    )
            else:
                if ephemeral_cluster and cluster_up:
                    if job_submitted:
                        logger.log(20, "Cluster will be destroyed after the job completes")
                    else:
                        self._tear_down_cluster(
                            cluster_manager,
                            key_name=key_name,
                            key_local_path=os.path.join(self.local_output_path, "utils"),
                        )

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

    def _get_image_uri(self, framework_version: str, instance_type: str, custom_image_uri: Optional[str] = None):
        image_uri = custom_image_uri
        if custom_image_uri is not None:
            framework_version, py_version = None, None
            logger.log(20, f"Training with custom_image_uri=={custom_image_uri}")
        else:
            framework_version, py_version = parse_framework_version(
                framework_version, "training", minimum_version="0.7.0"
            )
            logger.log(20, f"Training with framework_version=={framework_version}")
            image_uri = image_uris.retrieve(
                "autogluon",
                region=self.region,
                version=framework_version,
                py_version=py_version,
                image_scope="training",
                instance_type=instance_type,
            )
        return image_uri

    def _construct_ag_args(self, predictor_init_args, predictor_fit_args, **kwargs):
        config = dict(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            **kwargs,
        )
        return config

    def _upload_data(
        self, train_data: Union[str, pd.DataFrame], tune_data: Optional[Union[str, pd.DataFrame]] = None
    ) -> Tuple[str, str]:
        cloud_bucket, cloud_key_prefix = s3_path_to_bucket_prefix(self.cloud_output_path)
        util_key_prefix = cloud_key_prefix + "/utils"
        train_data = self._prepare_data(train_data, "train")
        logger.log(20, "Uploading train data..")
        upload_file(file_name=train_data, bucket=cloud_bucket, prefix=util_key_prefix)
        train_data = s3_bucket_prefix_to_path(
            bucket=cloud_bucket, prefix=f"{util_key_prefix}/{os.path.basename(train_data)}"
        )
        logger.log(20, "Train data uploaded successfully")
        if tune_data is not None:
            logger.log(20, "Uploading tune data...")
            tune_data = self._prepare_data(tune_data, "tune")
            upload_file(file_name=tune_data, bucket=cloud_bucket, prefix=util_key_prefix)
            tune_data = s3_bucket_prefix_to_path(
                bucket=cloud_bucket, prefix=f"{util_key_prefix}/{os.path.basename(tune_data)}"
            )
            logger.log(20, "Tune data uploaded successfully")
        return train_data, tune_data

    def _prepare_data(self, data: Union[str, pd.DataFrame], filename: str, output_type: str = "csv"):
        path = os.path.join(self.local_output_path, "utils")
        converter = FormatConverterFactory.get_converter(output_type)
        return converter.convert(data, path, filename)

    def _setup_role_and_permission(self):
        """
        Setup necessary role and permission to upload utils and launch up cluster
        """
        raise NotImplementedError

    def _setup_key(self, key_name: str, local_path: str) -> str:
        """
        Setup the ssh key required to connect to the cluster.

        Parameters
        ----------
        key_name: str
            Name of the key pair
        local_path: str
            Local path to store the private key

        Return
        ------
        str,
            Path to the local private key
        """
        raise NotImplementedError

    def _cleanup_key(self, key_name: str, local_path: Optional[str] = None):
        """
        Cleanup the ssh key required to connect to the cluster.

        Parameter
        ---------
        key_name: str
            Name of the key pair
        local_path: str, default = None
            Local path to the stored private key
        """
        raise NotImplementedError

    def _generate_config(
        self,
        config: Optional[Union[str, Dict[str, Any]]] = None,
        cluster_name: Optional[str] = None,
        instance_type: Optional[str] = None,
        instance_count: Optional[int] = None,
        volumes_size: Optional[int] = None,
        ami: Optional[str] = None,
        custom_image_uri: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
        initialization_commands: Optional[List[str]] = None,
    ):
        config_generator: RayClusterConfigGenerator = self._cluster_config_generator(
            config=config, cluster_name=cluster_name, region=self.region
        )
        if config is None:
            config_generator.update_config(
                instance_type=instance_type,
                instance_count=instance_count,
                volumes_size=volumes_size,
                custom_image_uri=custom_image_uri,
                ssh_key_path=ssh_key_path,
                head_instance_profile=get_instance_profile_arn(RAY_INSTANCE_PROFILE_NAME),
                worker_instance_profile=get_instance_profile_arn(RAY_INSTANCE_PROFILE_NAME),
                initialization_commands=initialization_commands,
            )
            config = os.path.join(self.local_output_path, "job", self._config_file_name)
            config_generator.save_config(save_path=config)
        return config

    def _tear_down_cluster(
        self, cluster_manager: RayClusterManager, key_name: Optional[str], key_local_path: Optional[str]
    ):
        logger.log(20, "Tearing down cluster")
        try:
            cluster_manager.down()
        except Exception as down_e:
            logger.warning("Failed to tear down the cluster. Please go to the console to terminate instanecs manually")
            raise down_e
        if key_name is not None:
            self._cleanup_key(key_name=key_name, local_path=key_local_path)
