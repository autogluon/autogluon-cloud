import copy
import json
import logging
import os
import tarfile
import tempfile
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
from botocore.exceptions import ClientError

from autogluon.common.loaders import load_pd
from autogluon.common.utils.s3_utils import is_s3_url, s3_path_to_bucket_prefix

from ..data import FormatConverterFactory
from ..job import SageMakerBatchTransformationJob, SageMakerFitJob
from ..scripts import ScriptManager
from ..utils.ag_sagemaker import (
    repack_model_with_serving_code,
    script_mode_environment,
    staged_serving_code,
    training_script_hyperparameters,
    upload_training_code,
)
from ..utils.aws_utils import resolve_execution_role, setup_sagemaker_session
from ..utils.constants import LOCAL_MODE, LOCAL_MODE_GPU, VALID_ACCEPT
from ..utils.deserializers import PandasDeserializer
from ..utils.dlc_utils import infer_sagemaker_ami_version, parse_framework_version, retrieve_image_uri
from ..utils.misc import MostRecentInsertedOrderedDict, sagemaker_timestamp, unique_name_from_base
from ..utils.sagemaker_api import (
    BATCH_PREDICT_OVERRIDE_KEYS,
    DEPLOY_OVERRIDE_KEYS,
    FIT_OVERRIDE_KEYS,
    check_override_keys,
    deep_merge,
    delete_endpoint,
    invoke_endpoint,
)
from ..utils.serializers import AutoGluonSerializationWrapper, AutoGluonSerializer
from ..utils.tag_utils import build_tags
from ..utils.utils import (
    convert_image_path_to_encoded_bytes_in_dataframe,
    is_image_file,
    split_pred_and_pred_proba,
    zipfolder,
)
from .backend import Backend
from .constant import SAGEMAKER

logger = logging.getLogger(__name__)

SAGEMAKER_MODEL_SERVER_WORKERS = "SAGEMAKER_MODEL_SERVER_WORKERS"
_SERVERLESS_CONFIG_FIELDS = {
    "memory_size_in_mb": "MemorySizeInMB",
    "max_concurrency": "MaxConcurrency",
    "provisioned_concurrency": "ProvisionedConcurrency",
}


def _reject_local_mode(instance_type: Optional[str]) -> None:
    if instance_type in (LOCAL_MODE, LOCAL_MODE_GPU):
        raise ValueError(
            f"instance_type={instance_type!r} (SageMaker local mode) is no longer supported. "
            "Use a SageMaker instance type such as 'ml.m5.2xlarge'."
        )


def _s3_channel(channel_name: str, s3_uri: str) -> Dict[str, Any]:
    return {
        "ChannelName": channel_name,
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": s3_uri,
                "S3DataDistributionType": "FullyReplicated",
            }
        },
    }


def _to_request_fields(settings: Dict[str, Any], fields: Dict[str, str], arg_name: str) -> Dict[str, Any]:
    """Rename the snake_case keys of a user-facing settings dict to the SageMaker API field names."""
    unknown = sorted(set(settings) - set(fields))
    if unknown:
        raise ValueError(f"Unsupported `{arg_name}` key(s) {unknown}. Valid keys: {list(fields)}.")
    return {fields[key]: value for key, value in settings.items()}


class SagemakerBackend(Backend):
    name = SAGEMAKER

    def __init__(
        self,
        local_output_path: str,
        cloud_output_path: str,
        predictor_type: str,
        role: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.initialize(
            local_output_path=local_output_path,
            cloud_output_path=cloud_output_path,
            predictor_type=predictor_type,
            role=role,
            **kwargs,
        )

    def initialize(self, role: Optional[str] = None, **kwargs) -> None:
        """Initialize the backend.

        Parameters
        ----------
        role
            SageMaker execution role ARN. See
            :func:`autogluon.cloud.utils.aws_utils.resolve_execution_role` for the resolution order.
        """
        super().initialize(**kwargs)
        self.sagemaker_session = setup_sagemaker_session()
        try:
            self.role_arn = resolve_execution_role(role, backend_name=SAGEMAKER, session=self.sagemaker_session)
        except ClientError as e:
            logger.warning(
                "Failed to resolve SageMaker execution role. Pass `role=<arn>` to the predictor/model "
                "or run `autogluon.cloud.bootstrap()` / `register()` to persist one."
            )
            raise e
        self._region = self.sagemaker_session.boto_region_name
        self._fit_job: SageMakerFitJob = SageMakerFitJob(session=self.sagemaker_session)
        self._batch_transform_jobs = MostRecentInsertedOrderedDict()

    def _realtime_serializer(self):
        """Serializer used for realtime endpoint requests"""
        return AutoGluonSerializer()

    def _resolve_tags(self, extra_tags: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """Tags for a created SageMaker resource: default + extra tags."""
        return build_tags(self.predictor_type, extra_tags=extra_tags)

    def attach_job(self, job_name: str) -> None:
        """
        Attach to a existing training job.
        This is useful when the local process crashed and you want to reattach to the previous job

        Parameters
        ----------
        job_name: str
            The name of the job being attached
        """
        self._fit_job = SageMakerFitJob.attach(job_name, session=self.sagemaker_session)

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
        data_channels: Dict[str, Optional[Union[str, pd.DataFrame]]],
        image_column: Optional[str] = None,
        leaderboard: bool = True,
        framework_version: str = "latest",
        job_name: Optional[str] = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: Union[int, str] = 1,
        volume_size: int = 256,
        custom_image_uri: Optional[str] = None,
        timeout: int = 24 * 60 * 60,
        wait: bool = True,
        backend_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        extra_ag_args: Optional[Dict[str, Any]] = None,
        extra_tags: Optional[List[Dict[str, str]]] = None,
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
            Fit args for the predictor (must NOT contain data inputs — pass those via ``data_channels``).
        data_channels: Dict[str, Union[str, pd.DataFrame, None]]
            Mapping from data-input name to a DataFrame or local/S3 path. Each non-None entry is uploaded
            as a separate SageMaker channel; the train script reads it via ``SM_CHANNEL_<KEY_UPPER>``.
            Must contain a non-None ``train_data`` entry; subclasses define which additional keys are honored.
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
            If None, AutoGluon Cloud creates one with a predictor- or model-specific prefix.
        instance_type: str, default = 'ml.m5.2xlarge'
            Instance type the predictor will be trained on with SageMaker.
        instance_count: int, default = 1
            Number of instance used to fit the predictor.
        volume_size: int, default = 256
            Size in GB of the EBS volume to use for storing input data during training (default: 256).
            Must be large enough to store training data if File Mode is used (which is the default).
        timeout: int, default = 24*60*60
            Timeout in seconds for training. This timeout doesn't include time for pre-processing or launching up the training job.
        wait: bool, default = True
            Whether the call should wait until the job completes
            To be noticed, the function won't return immediately because there are some preparations needed prior fit.
            Use `get_fit_job_status` to get job status.
        backend_overrides: Optional[Dict[str, Dict[str, Any]]], default = None
            Raw ``CreateTrainingJob`` request fields (SageMaker API / boto3 PascalCase names) under the
            ``"create_training_job"`` key, deep-merged over the request built by AutoGluon-Cloud.
        extra_ag_args: Optional[Dict[str, Any]], default = None
            Additional entries to merge into ``ag_args.json``. Use this to ship caller-specific metadata to the
            train script (e.g. ``predict_after_fit``, ``save_predictor``, or ``id_column`` /
            ``timestamp_column`` for time series).
        """
        if data_channels.get("train_data") is None:
            raise ValueError("`data_channels['train_data']` is required.")
        _reject_local_mode(instance_type)
        overrides = check_override_keys(backend_overrides, FIT_OVERRIDE_KEYS)
        predictor_fit_args = copy.deepcopy(predictor_fit_args)
        # Resolve any path inputs (str or pathlib.Path) into DataFrames so they can be CSV-uploaded as SageMaker channels.
        data_channels = {
            k: load_pd.load(str(v)) if isinstance(v, (str, os.PathLike)) else v
            for k, v in data_channels.items()
            if v is not None
        }
        if custom_image_uri:
            framework_version, py_version = None, None
            logger.log(20, f"Training with custom_image_uri=={custom_image_uri}")
        else:
            framework_version, py_version = parse_framework_version(
                framework_version, "training", minimum_version="0.6.0"
            )
            logger.log(20, f"Training with framework_version=={framework_version}")

        if not job_name:
            job_name = unique_name_from_base(self.resource_prefix)

        if instance_count == "auto":
            instance_count = 1
        if instance_count > 1:
            logger.warning(
                "We don't support distributed training with sagemaker backend yet. Will change instance_count to be 1"
            )
            instance_count = 1

        self._train_script_path = ScriptManager.get_train_script(
            backend_type=self.name, framework_version=framework_version
        )
        entry_point = self._train_script_path

        ag_args = dict(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            leaderboard=leaderboard,
        )
        # Get the label from predictor_init_args
        label = predictor_init_args.get("label") or predictor_init_args.get("target") or None
        if image_column is not None:
            ag_args["image_column"] = image_column
        if extra_ag_args:
            ag_args.update(extra_ag_args)
        if ag_args.get("predict_after_fit"):
            predictions_path = ag_args.get("predictions_path")
            if predictions_path is None:
                ag_args["predictions_path"] = f"{self.cloud_output_path}/{job_name}/predictions.csv"
            elif not is_s3_url(predictions_path) or not predictions_path.endswith((".csv", ".parquet")):
                raise ValueError(
                    f"`predictions_path` must be a full S3 URL ending in '.csv' or '.parquet' "
                    f"(e.g. 's3://bucket/key/predictions.parquet'), got {predictions_path!r}."
                )
        ag_args_path = os.path.join(self.local_output_path, "utils", "ag_args.json")
        self.prepare_args(path=ag_args_path, **ag_args)
        inputs = self._upload_fit_artifact(
            data_channels=data_channels,
            label=label,
            ag_args=ag_args_path,
            image_column=image_column,
            serving_script=ScriptManager.get_serve_script(
                backend_type=self.name, framework_version=framework_version
            ),  # Training and Inference should have the same framework_version
        )
        code_uri = upload_training_code(
            entry_point=entry_point,
            sagemaker_session=self.sagemaker_session,
            s3_uri_prefix=f"{self.cloud_output_path}/code/{job_name}/source",
        )

        request: Dict[str, Any] = {
            "TrainingJobName": job_name,
            "RoleArn": self.role_arn,
            "AlgorithmSpecification": {
                "TrainingImage": custom_image_uri
                or retrieve_image_uri(framework_version, self._region, "training", instance_type, py_version),
                "TrainingInputMode": "File",
            },
            "HyperParameters": training_script_hyperparameters(
                entry_point=entry_point, submit_directory=code_uri, job_name=job_name, region=self._region
            ),
            "InputDataConfig": [_s3_channel(name, uri) for name, uri in inputs.items()],
            "OutputDataConfig": {"S3OutputPath": self.cloud_output_path + "/model"},
            "ResourceConfig": {
                "InstanceType": instance_type,
                "InstanceCount": instance_count,
                "VolumeSizeInGB": volume_size,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": timeout},
            "ProfilerConfig": {"DisableProfiler": True},
            "Tags": self._resolve_tags(extra_tags),
        }
        request = deep_merge(request, overrides.get("create_training_job", {}))

        self._fit_job = SageMakerFitJob(session=self.sagemaker_session)
        self._fit_job.run(training_job_request=request, framework_version=framework_version, wait=wait)

    def _create_model(
        self,
        model_name: str,
        model_data: str,
        image_uri: str,
        entry_point: str,
        environment: Dict[str, str],
        tags: List[Dict[str, str]],
        overrides: Dict[str, Dict[str, Any]],
    ) -> str:
        """Create a SageMaker model serving ``model_data`` (which must contain the serving code under ``code/``)."""
        # PYTHONUNBUFFERED disables output buffering for endpoint logging.
        container_environment = {
            "PYTHONUNBUFFERED": "1",
            **environment,
            **script_mode_environment(entry_point, self._region),
        }
        request: Dict[str, Any] = {
            "ModelName": model_name,
            "PrimaryContainer": {
                "Image": image_uri,
                "ModelDataUrl": model_data,
                "Environment": container_environment,
            },
            "ExecutionRoleArn": self.role_arn,
            "Tags": tags,
        }
        request = deep_merge(request, overrides.get("create_model", {}))
        logger.log(20, "Creating inference model...")
        self.sagemaker_session.sagemaker_client.create_model(**request)
        logger.log(20, "Inference model created successfully")
        return request["ModelName"]

    def _prepare_model_data(
        self,
        predictor_path: str,
        entry_point: str,
        repack: bool,
        repacked_model_uri: str,
    ) -> str:
        """Return an S3 model tarball that contains the serving code, repacking ``predictor_path`` if needed."""
        if not repack:
            return predictor_path
        logger.log(20, "Repacking the serving code into the model artifact...")
        return repack_model_with_serving_code(
            model_data=predictor_path,
            entry_point=entry_point,
            repacked_model_uri=repacked_model_uri,
            sagemaker_session=self.sagemaker_session,
        )

    def deploy(
        self,
        predictor_path: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        framework_version: str = "latest",
        instance_type: Optional[str] = "ml.m5.2xlarge",
        initial_instance_count: int = 1,
        custom_image_uri: Optional[str] = None,
        volume_size: Optional[int] = None,
        wait: bool = True,
        backend_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        entry_point: Optional[str] = None,
        fm_serve_config: Optional[Dict[str, Any]] = None,
        inference_mode: Literal["realtime", "serverless"] = "realtime",
        inference_config: Optional[Dict[str, Any]] = None,
        repack: bool = True,
        extra_tags: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """
        Deploy a predictor as a SageMaker endpoint, which can be used to do real-time inference later.
        This method creates a SageMaker model with the trained predictor, an endpoint config, and the endpoint.

        Parameters
        ----------
        predictor_path: str
            Path to the predictor tarball you want to deploy.
            Path can be both a local path or a S3 location.
            If None, will deploy the most recent trained predictor trained with `fit()`.
        endpoint_name: str
            The endpoint name to use for the deployment.
            If None, AutoGluon Cloud creates one with a predictor- or model-specific prefix.
        framework_version: str, default = `latest`
            Inference container version of autogluon.
            If `latest`, will use the latest available container version.
            If provided a specific version, will use this version.
            If `custom_image_uri` is set, this argument will be ignored.
        instance_type: str, default = 'ml.m5.2xlarge'
            Instance to be deployed for the endpoint
        initial_instance_count: int, default = 1,
            Initial number of instances to be deployed for the endpoint
        custom_image_uri: Optional[str], default = None,
            Custom image to use to deploy endpoint with.
            If not specified, with use official DLC image:
            https://aws.github.io/deep-learning-containers/reference/available_images/#autogluon
        volume_size: int, default = None
           The size, in GB, of the ML storage volume attached to individual inference instance associated with the production variant.
           Currenly only Amazon EBS gp2 storage volumes are supported.
        wait: Bool, default = True,
            Whether to wait for the endpoint to be deployed.
            To be noticed, the function won't return immediately because there are some preparations needed prior deployment.
        backend_overrides: Optional[Dict[str, Dict[str, Any]]], default = None
            Raw request fields (SageMaker API / boto3 PascalCase names) deep-merged over the requests built by
            AutoGluon-Cloud. Valid keys: ``"create_model"``, ``"production_variant"``,
            ``"create_endpoint_config"``, ``"create_endpoint"``.
        entry_point: Optional[str], default = None
            Serve script to use instead of the predictor type's default.
        fm_serve_config: Optional[Dict[str, Any]], default = None
            Configuration dict passed to the FM serve script via the AG_FM_SERVE_CONFIG env var.
        inference_mode: {"realtime", "serverless"}, default = "realtime"
            Endpoint type. ``"serverless"`` provisions a SageMaker Serverless Inference endpoint
            (no instance management, scales to zero).
        inference_config: Optional[Dict[str, Any]], default = None
            Serverless overrides forwarded to the production variant's ``ServerlessConfig``
            (``memory_size_in_mb``, ``max_concurrency``, ``provisioned_concurrency``).
        repack: bool, default = True
            Whether to download ``predictor_path``, inject the serve script, and re-upload it. Set to False when
            ``predictor_path`` already contains the serve script (e.g. an artifact bundled by
            :meth:`FoundationModel.cache_model_artifact`) to skip the round-trip. Ignored when ``predictor_path`` is
            None.
        """
        assert self.endpoint_name is None, (
            "There is an endpoint already attached. Either detach it with `detach` or clean it up with `cleanup_deployment`"
        )
        overrides = check_override_keys(backend_overrides, DEPLOY_OVERRIDE_KEYS)
        serverless_config = None
        if inference_mode == "serverless":
            preset = {"memory_size_in_mb": 4096, "max_concurrency": 5}
            serverless_config = _to_request_fields(
                {**preset, **(inference_config or {})}, _SERVERLESS_CONFIG_FIELDS, "inference_config"
            )
        if inference_mode == "serverless" and instance_type is None:
            # Needed to infer the container image (CPU vs GPU) downstream — serverless is CPU-only.
            instance_type = "ml.m5.2xlarge"
        _reject_local_mode(instance_type)
        if not endpoint_name:
            endpoint_name = unique_name_from_base(self.resource_prefix)

        # Resolve container image
        if custom_image_uri:
            framework_version, py_version = None, None
            logger.log(20, f"Deploying with custom_image_uri=={custom_image_uri}")
        else:
            framework_version, py_version = parse_framework_version(
                framework_version, "inference", minimum_version="0.6.0"
            )
            logger.log(20, f"Deploying with framework_version=={framework_version}")

        if volume_size and instance_type.startswith(("ml.p", "ml.g")):
            logger.warning(
                f"SageMaker backend doesn't support providing custom volume_size. Specified {volume_size} GB. Will ignore."
            )
            volume_size = None

        # Resolve model artifact:
        # - predictor_path provided → use as-is
        # - predictor_path=None, fit job exists, non-FM deploy → use fit output
        # - predictor_path=None, no fit job (or FM deploy) → no artifact
        if predictor_path is None and self._fit_job is not None and fm_serve_config is None:
            predictor_path = self._fit_job.get_output_path()
        if predictor_path:
            predictor_path = self._upload_predictor(predictor_path, f"endpoints/{endpoint_name}/predictor")

        user_entry_point = entry_point
        if entry_point is None:
            self._serve_script_path = ScriptManager.get_serve_script(
                backend_type=self.name, framework_version=framework_version
            )
            entry_point = self._serve_script_path

        # Decide whether the tarball already contains the entry_point script — if yes, use it as-is;
        # if no, repack the script into it.
        if predictor_path is None:
            model_data = self._create_serve_script_tarball(entry_point, endpoint_name)
        else:
            is_default_fit_output = (
                self._fit_job is not None
                and predictor_path == self._fit_job.get_output_path()
                and user_entry_point is None
            )
            model_data = self._prepare_model_data(
                predictor_path,
                entry_point=entry_point,
                repack=repack and not is_default_fit_output,
                repacked_model_uri=f"{self.cloud_output_path}/endpoints/{endpoint_name}/model/model.tar.gz",
            )

        container_environment = {SAGEMAKER_MODEL_SERVER_WORKERS: "1"}
        if fm_serve_config is not None:
            container_environment["AG_FM_SERVE_CONFIG"] = json.dumps(fm_serve_config)
        if inference_mode == "serverless":
            # Serverless containers run with `/` as cwd and a read-only root, so TorchServe's
            # default `logs/` path resolves to `/logs` and startup fails. Redirect to /tmp.
            container_environment.setdefault("LOG_LOCATION", "/tmp")
            container_environment.setdefault("METRICS_LOCATION", "/tmp")

        tags = self._resolve_tags(extra_tags)
        model_name = self._create_model(
            model_name=unique_name_from_base(endpoint_name),
            model_data=model_data,
            image_uri=custom_image_uri
            or retrieve_image_uri(framework_version, self._region, "inference", instance_type, py_version),
            entry_point=entry_point,
            environment=container_environment,
            tags=tags,
            overrides=overrides,
        )

        variant: Dict[str, Any] = {"VariantName": "AllTraffic", "ModelName": model_name}
        if inference_mode == "realtime":
            variant["InstanceType"] = instance_type
            variant["InitialInstanceCount"] = initial_instance_count
            if volume_size:
                variant["VolumeSizeInGB"] = volume_size
            inference_ami_version = infer_sagemaker_ami_version(
                custom_image_uri, instance_type, image_scope="inference"
            )
            if inference_ami_version is not None:
                variant["InferenceAmiVersion"] = inference_ami_version
        elif inference_mode == "serverless":
            variant["ServerlessConfig"] = serverless_config
        else:
            raise ValueError(f"Unsupported inference_mode={inference_mode!r}")
        variant = deep_merge(variant, overrides.get("production_variant", {}))

        endpoint_config_request: Dict[str, Any] = {
            "EndpointConfigName": endpoint_name,
            "ProductionVariants": [variant],
            "Tags": tags,
        }
        endpoint_config_request = deep_merge(endpoint_config_request, overrides.get("create_endpoint_config", {}))
        endpoint_request = deep_merge(
            {
                "EndpointName": endpoint_name,
                "EndpointConfigName": endpoint_config_request["EndpointConfigName"],
                "Tags": tags,
            },
            overrides.get("create_endpoint", {}),
        )

        logger.log(20, f"Deploying model to the endpoint (inference_mode={inference_mode})")
        client = self.sagemaker_session.sagemaker_client
        client.create_endpoint_config(**endpoint_config_request)
        client.create_endpoint(**endpoint_request)
        self.endpoint_name = endpoint_request["EndpointName"]
        if wait:
            client.get_waiter("endpoint_in_service").wait(EndpointName=self.endpoint_name)

    def _create_serve_script_tarball(self, serve_script_path: str, endpoint_name: str) -> str:
        """Create a minimal model.tar.gz containing the serve script + serving_utils/ under code/."""

        tarball_dir = tempfile.mkdtemp(prefix="ag_serve_")
        tarball_path = os.path.join(tarball_dir, "model.tar.gz")
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(serve_script_path, arcname=f"code/{os.path.basename(serve_script_path)}")
            tar.add(ScriptManager.SAGEMAKER_SERVING_UTILS_DIR, arcname="code/serving_utils")
        s3_key = f"endpoints/{endpoint_name}/model/model.tar.gz"
        s3_path = self._upload_predictor(tarball_path, s3_key)
        return s3_path

    def cleanup_deployment(self) -> None:
        """
        Delete endpoint, endpoint configuration and deployed model
        """
        assert self.endpoint_name is not None, "No deployed endpoint detected"
        delete_endpoint(self.endpoint_name, self.sagemaker_session)
        self.endpoint_name = None

    def attach_endpoint(self, endpoint: str) -> None:
        """
        Attach the current backend to an existing SageMaker endpoint.

        Parameters
        ----------
        endpoint: str
            Name of the endpoint being attached to.
        """
        assert self.endpoint_name is None, (
            "There is an endpoint already attached. Either detach it with `detach` or clean it up with `cleanup_deployment`"
        )
        if not isinstance(endpoint, str):
            raise ValueError(f"Please provide the endpoint name as a string, got {type(endpoint).__name__}.")
        self.endpoint_name = endpoint

    def detach_endpoint(self) -> str:
        """Detach the current endpoint and return its name"""
        assert self.endpoint_name is not None, "There is no attached endpoint"
        detached_endpoint = self.endpoint_name
        self.endpoint_name = None
        return detached_endpoint

    def predict_real_time(
        self,
        test_data: Union[str, pd.DataFrame],
        test_data_image_column: Optional[str] = None,
        accept: str = "application/x-parquet",
        inference_kwargs: Optional[Dict[str, Any]] = None,
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
        inference_kwargs: Optional[Dict[str, Any]], default = None
            Additional args that you would pass to `predict` calls of an AutoGluon logic

        Returns
        -------
        Pandas.Series
        Predict results in Series
        """
        self._validate_predict_real_time_args(accept)
        test_data = self._load_predict_real_time_test_data(test_data, test_data_image_column=test_data_image_column)
        pred, _ = self._predict_real_time(test_data=test_data, accept=accept, inference_kwargs=inference_kwargs)

        return pred

    def predict_proba_real_time(
        self,
        test_data: Union[str, pd.DataFrame],
        test_data_image_column: Optional[str] = None,
        accept: str = "application/x-parquet",
        inference_kwargs: Optional[Dict[str, Any]] = None,
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
        inference_kwargs: Optional[Dict[str, Any]], default = None
            Additional args that you would pass to `predict` calls of an AutoGluon logic

        Returns
        -------
        Pandas.DataFrame or Pandas.Series
            Will return a Pandas.Series when it's a regression problem. Will return a Pandas.DataFrame otherwise
        """
        self._validate_predict_real_time_args(accept)
        test_data = self._load_predict_real_time_test_data(test_data, test_data_image_column=test_data_image_column)
        pred, proba = self._predict_real_time(test_data=test_data, accept=accept, inference_kwargs=inference_kwargs)

        if proba is None:
            return pred

        return proba

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
        predictions_path: Optional[str] = None,
        backend_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[pd.Series]:
        """
        Predict using SageMaker batch transform.
        When minimizing latency isn't a concern, then the batch transform functionality may be easier, more scalable, and more appropriate.
        If you want to minimize latency, use `predict_real_time()` instead.
        To learn more: https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html
        This method creates a SageMaker model with the trained predictor and runs a transform job with it.

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
            If None, AutoGluon Cloud creates one with a predictor- or model-specific prefix.
        instance_count: int, default = 1,
            Number of instances used to do batch transform.
        instance_type: str, default = 'ml.m5.2xlarge'
            Instance to be used for batch transform.
        wait: bool, default = True
            Whether to wait for batch transform to complete.
            To be noticed, the function won't return immediately because there are some preparations needed prior transform.
        predictions_path: Optional[str], default = None
            S3 prefix under which the batch transform job writes its results (``<predictions_path>/<input file>.out``).
            Defaults to ``{cloud_output_path}/batch_transform/<timestamp>/results``.
        backend_overrides: Optional[Dict[str, Dict[str, Any]]], default = None
            Raw request fields (SageMaker API / boto3 PascalCase names) deep-merged over the requests built by
            AutoGluon-Cloud. Valid keys: ``"create_model"``, ``"create_transform_job"``.

        Returns
        -------
        Optional Pandas.Series
            Predict results in Series if `wait` is True
            None if `wait` is False
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
            predictions_path=predictions_path,
            backend_overrides=backend_overrides,
            original_features=self.original_features,
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
        predictions_path: Optional[str] = None,
        backend_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[Union[Tuple[pd.Series, Union[pd.DataFrame, pd.Series]], Union[pd.DataFrame, pd.Series]]]:
        """
        Predict using SageMaker batch transform.
        When minimizing latency isn't a concern, then the batch transform functionality may be easier, more scalable, and more appropriate.
        If you want to minimize latency, use `predict_real_time()` instead.
        To learn more: https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html
        This method creates a SageMaker model with the trained predictor and runs a transform job with it.

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
            If None, AutoGluon Cloud creates one with a predictor- or model-specific prefix.
        instance_count: int, default = 1,
            Number of instances used to do batch transform.
        instance_type: str, default = 'ml.m5.2xlarge'
            Instance to be used for batch transform.
        wait: bool, default = True
            Whether to wait for batch transform to complete.
            To be noticed, the function won't return immediately because there are some preparations needed prior transform.
        predictions_path: Optional[str], default = None
            S3 prefix under which the batch transform job writes its results (``<predictions_path>/<input file>.out``).
            Defaults to ``{cloud_output_path}/batch_transform/<timestamp>/results``.
        backend_overrides: Optional[Dict[str, Dict[str, Any]]], default = None
            Raw request fields (SageMaker API / boto3 PascalCase names) deep-merged over the requests built by
            AutoGluon-Cloud. Valid keys: ``"create_model"``, ``"create_transform_job"``.


        Returns
        -------
        Optional[Union[Tuple[pd.Series, Union[pd.DataFrame, pd.Series]], Union[pd.DataFrame, pd.Series]]]
            If `wait` is False, will return None or (None, None) if `include_predict` is True
            If `wait` is True and `include_predict` is True,
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
            predictions_path=predictions_path,
            backend_overrides=backend_overrides,
            original_features=self.original_features,
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

    def get_fit_predict_results(self) -> pd.DataFrame:
        """Read predictions produced by a completed ``fit_predict`` job from S3."""
        ag_args = self._download_ag_args_from_job()
        predictions_path = ag_args.get("predictions_path")
        assert predictions_path is not None, "No fit_predict job found. Call `fit_predict()` first."
        bucket, key = s3_path_to_bucket_prefix(predictions_path)
        with tempfile.TemporaryDirectory(prefix="ag_fit_predict_") as tmpdir:
            local_path = os.path.join(tmpdir, os.path.basename(key))
            self.sagemaker_session.s3_client.download_file(bucket, key, local_path)
            return load_pd.load(local_path)

    def _download_ag_args_from_job(self) -> Dict[str, Any]:
        """Fetch and parse the ``ag_args.json`` that was uploaded as the ``ag_args`` channel.

        Each training job carries the exact config it was launched with as an input channel,
        making this the authoritative source — independent of local-disk lifetime.
        """
        job_name = self._fit_job.job_name
        assert job_name is not None, "No fit job found. Call `fit()` / `fit_predict()` first."
        desc = self.sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=job_name)
        channels = {c["ChannelName"]: c["DataSource"]["S3DataSource"]["S3Uri"] for c in desc["InputDataConfig"]}
        ag_args_uri = channels.get("ag_args")
        assert ag_args_uri is not None, (
            f"Training job {job_name!r} has no `ag_args` input channel — cannot recover predictions_path."
        )
        bucket, key = s3_path_to_bucket_prefix(ag_args_uri)
        assert key.endswith(".json"), f"Expected ag_args channel to point to a .json file, got {ag_args_uri!r}"
        with tempfile.TemporaryDirectory(prefix="ag_args_") as tmpdir:
            local_path = os.path.join(tmpdir, os.path.basename(key))
            self.sagemaker_session.s3_client.download_file(bucket, key, local_path)
            with open(local_path, "r") as f:
                return json.load(f)

    def _construct_ag_args(self, predictor_init_args, predictor_fit_args, leaderboard, **kwargs):
        config = dict(
            predictor_type=self.predictor_type,
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            leaderboard=leaderboard,
            **kwargs,
        )
        return config

    def _prepare_data(self, data, filename, output_type="csv"):
        path = os.path.join(self.local_output_path, "utils")
        converter = FormatConverterFactory.get_converter(output_type)
        return converter.convert(data, path, filename)

    def _prepare_and_upload_data(self, data, filename, bucket, key_prefix, output_type="csv"):
        """Convert ``data`` to a local CSV via ``_prepare_data`` and upload it under ``key_prefix``.

        Returns the S3 URI of the uploaded file.
        """
        local_path = self._prepare_data(data, filename, output_type=output_type)
        logger.log(20, f"Uploading {filename} data...")
        return self.sagemaker_session.upload_data(path=local_path, bucket=bucket, key_prefix=key_prefix)

    def _find_common_path_and_replace_image_column(self, data, image_column):
        common_path = os.path.commonpath(data[image_column].tolist())
        common_path_head = os.path.split(common_path)[0]  # we keep the base dir to match zipping behavior
        data = data.assign(
            **{image_column: data[image_column].apply(lambda path: os.path.relpath(path, common_path_head))}
        )
        return data, common_path

    def _upload_fit_artifact(
        self,
        data_channels,
        label,
        ag_args,
        serving_script,
        image_column=None,
    ):
        cloud_bucket, cloud_key_prefix = s3_path_to_bucket_prefix(self.cloud_output_path)
        util_key_prefix = cloud_key_prefix + "/utils"

        # Image-column mode: rewrite image paths to be container-relative; common image directories
        # are zipped and uploaded as separate train_images / tune_images channels below.
        common_train_data_path, common_tune_data_path = None, None
        if image_column is not None:
            data_channels["train_data"], common_train_data_path = self._find_common_path_and_replace_image_column(
                data=data_channels["train_data"], image_column=image_column
            )
            if data_channels.get("tuning_data") is not None:
                data_channels["tuning_data"], common_tune_data_path = self._find_common_path_and_replace_image_column(
                    data=data_channels["tuning_data"], image_column=image_column
                )

        self.original_features = [col for col in data_channels["train_data"].columns if col != label]

        inputs: Dict[str, str] = {}
        for channel_name, channel_data in data_channels.items():
            inputs[channel_name] = self._prepare_and_upload_data(
                channel_data, channel_name, cloud_bucket, util_key_prefix
            )
            logger.log(20, f"{channel_name} uploaded successfully")

        inputs["ag_args"] = self.sagemaker_session.upload_data(
            path=ag_args, bucket=cloud_bucket, key_prefix=util_key_prefix
        )
        inputs["serving"] = self._upload_serving_files(
            entry_point=serving_script, bucket=cloud_bucket, key_prefix=util_key_prefix
        )

        train_images_input = self._upload_fit_image_artifact(
            image_dir_path=common_train_data_path, bucket=cloud_bucket, key_prefix=util_key_prefix
        )
        tune_images_input = self._upload_fit_image_artifact(
            image_dir_path=common_tune_data_path, bucket=cloud_bucket, key_prefix=util_key_prefix
        )
        if train_images_input is not None:
            inputs["train_images"] = train_images_input
        if tune_images_input is not None:
            inputs["tune_images"] = tune_images_input

        return inputs

    def _upload_serving_files(self, entry_point: str, bucket: str, key_prefix: str) -> str:
        with staged_serving_code(entry_point) as staging_dir:
            return self.sagemaker_session.upload_data(
                path=staging_dir, bucket=bucket, key_prefix=key_prefix + "/serving"
            )

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
        assert self.endpoint_name is not None, "Please call `deploy()` to deploy an endpoint first."
        assert accept in VALID_ACCEPT, f"Invalid accept type: {accept}. Options are {VALID_ACCEPT}."

    def _load_predict_real_time_test_data(self, test_data, test_data_image_column):
        if isinstance(test_data, str):
            test_data = load_pd.load(test_data)
        if isinstance(test_data, pd.DataFrame):
            if test_data_image_column is not None:
                test_data = convert_image_path_to_encoded_bytes_in_dataframe(test_data, test_data_image_column)

        return test_data

    def _predict_real_time(self, test_data, accept, split_pred_proba=True, inference_kwargs=None, content_type=None):
        try:
            if not isinstance(test_data, AutoGluonSerializationWrapper):
                test_data = AutoGluonSerializationWrapper(data=test_data, inference_kwargs=inference_kwargs)
            prediction = invoke_endpoint(
                self.endpoint_name,
                self.sagemaker_session,
                test_data,
                serializer=self._realtime_serializer(),
                deserializer=PandasDeserializer(),
                content_type=content_type,
                accept=accept,
            )
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
        predictions_path=None,
        backend_overrides=None,
        split_pred_proba=True,
        original_features=None,
        content_type="text/csv",
        split_type="Line",
        accept="application/json",
        assemble_with="Line",
        batch_strategy="MultiRecord",
    ):
        _reject_local_mode(instance_type)
        overrides = check_override_keys(backend_overrides, BATCH_PREDICT_OVERRIDE_KEYS)
        if predictions_path is not None and not is_s3_url(predictions_path):
            raise ValueError(f"`predictions_path` must be an S3 URL, got {predictions_path!r}.")
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

        output_path = self.cloud_output_path + "/batch_transform" + f"/{sagemaker_timestamp()}"

        cloud_bucket, cloud_key_prefix = s3_path_to_bucket_prefix(output_path)
        logger.log(20, "Preparing autogluon predictor...")
        predictor_path = self._upload_predictor(predictor_path, cloud_key_prefix + "/predictor")

        if not job_name:
            job_name = unique_name_from_base(self.resource_prefix)

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

        # If a directory of images, upload directly
        if isinstance(test_data, str) and not os.path.isdir(test_data):
            # either a file to a dataframe, or a file to an image
            if is_image_file(test_data):
                logger.warning(
                    "Are you sure you want to do batch inference on a single image? You might want to try `deploy()` and `predict_real_time()` instead"
                )
            elif original_features is not None:
                # Loading is only needed for the column check below — skip it for predictors that don't track
                # `original_features` (e.g. timeseries) to avoid loading the file as a DataFrame just to upload it.
                test_data = load_pd.load(test_data)

        if isinstance(test_data, pd.DataFrame) and original_features is not None:
            expected_columns = original_features
            incoming_columns = test_data.columns.tolist()
            missing_columns = set(expected_columns) - set(incoming_columns)
            if missing_columns:
                raise ValueError(f"Missing columns in input data: {missing_columns}")
            # Remove extra columns and reorder to match the model's expected order
            test_data = test_data[expected_columns]

        test_input = self._upload_batch_predict_data(test_data, cloud_bucket, cloud_key_prefix)

        self._serve_script_path = ScriptManager.get_serve_script(
            backend_type=self.name, framework_version=framework_version
        )
        entry_point = self._serve_script_path
        # Models not produced by this predictor's fit job don't carry our serving code yet.
        repack = predictor_path != self._fit_job.get_output_path()
        model_data = self._prepare_model_data(
            predictor_path,
            entry_point=entry_point,
            repack=repack,
            repacked_model_uri=f"{output_path}/model/model.tar.gz",
        )

        tags = self._resolve_tags()
        model_name = self._create_model(
            model_name=job_name,
            model_data=model_data,
            image_uri=custom_image_uri
            or retrieve_image_uri(framework_version, self._region, "inference", instance_type, py_version),
            entry_point=entry_point,
            environment={},
            tags=tags,
            overrides=overrides,
        )

        transform_input: Dict[str, Any] = {
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": test_input}},
            "ContentType": content_type,
        }
        if split_type is not None:
            transform_input["SplitType"] = split_type
        transform_output: Dict[str, Any] = {
            "S3OutputPath": (predictions_path or output_path + "/results").rstrip("/"),
            "Accept": accept,
        }
        if assemble_with is not None:
            transform_output["AssembleWith"] = assemble_with
        transform_resources: Dict[str, Any] = {"InstanceType": instance_type, "InstanceCount": instance_count}
        transform_ami_version = infer_sagemaker_ami_version(custom_image_uri, instance_type, image_scope="transform")
        if transform_ami_version is not None:
            transform_resources["TransformAmiVersion"] = transform_ami_version
        request = {
            "TransformJobName": job_name,
            "ModelName": model_name,
            "TransformInput": transform_input,
            "TransformOutput": transform_output,
            "TransformResources": transform_resources,
            "BatchStrategy": batch_strategy,
            # Maximum size in MB of a single request to the container; larger inputs are split into multiple batches.
            "MaxPayloadInMB": 6,
            # The maximum number of HTTP requests made to each individual transform container at one time.
            "MaxConcurrentTransforms": 1,
            "Tags": tags,
        }
        request = deep_merge(request, overrides.get("create_transform_job", {}))

        batch_transform_job = SageMakerBatchTransformationJob(session=self.sagemaker_session)
        batch_transform_job.run(transform_job_request=request, wait=wait)
        self._batch_transform_jobs[job_name] = batch_transform_job

        pred, pred_proba = None, None
        if wait:
            bucket, key = s3_path_to_bucket_prefix(batch_transform_job.get_output_path())
            with tempfile.TemporaryDirectory(prefix="ag_batch_results_") as tmpdir:
                results_path = os.path.join(tmpdir, os.path.basename(key))
                self.sagemaker_session.s3_client.download_file(bucket, key, results_path)
                accept = request["TransformOutput"].get("Accept")
                if accept == "application/x-parquet":
                    results = pd.read_parquet(results_path)
                elif accept == "text/csv":
                    results = pd.read_csv(results_path)
                elif accept == "application/json":
                    results = pd.read_json(results_path)
                else:
                    raise ValueError(f"Unsupported accept type for batch inference results: {accept!r}")
            pred = results
            if split_pred_proba:
                pred, pred_proba = split_pred_and_pred_proba(results)

        return pred, pred_proba

    def __getstate__(self) -> Dict[str, Any]:
        """Custom implementation of the pickle process"""
        d = self.__dict__.copy()
        d["sagemaker_session"] = None
        d["_region"] = None
        return d

    def __setstate__(self, state):
        """Custom implementation of the unpickle process"""
        self.__dict__.update(state)
        self.sagemaker_session = setup_sagemaker_session()
        self._region = self.sagemaker_session.boto_region_name
        self._fit_job.session = self.sagemaker_session
        for job in self._batch_transform_jobs.values():
            job.session = self.sagemaker_session
