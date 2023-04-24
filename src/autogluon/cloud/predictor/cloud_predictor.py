from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

import boto3
import pandas as pd
import sagemaker

from autogluon.common.loaders import load_pkl
from autogluon.common.savers import save_pkl
from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.common.utils.s3_utils import is_s3_url, s3_path_to_bucket_prefix
from autogluon.common.utils.utils import setup_outputdir

from ..backend.backend import Backend
from ..backend.backend_factory import BackendFactory
from ..backend.constant import SAGEMAKER
from ..endpoint.endpoint import Endpoint
from ..utils.utils import unzip_file

logger = logging.getLogger(__name__)


class CloudPredictor(ABC):
    predictor_file_name = "CloudPredictor.pkl"

    def __init__(
        self,
        cloud_output_path: str,
        local_output_path: Optional[str] = None,
        backend: str = SAGEMAKER,
        verbosity: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        cloud_output_path: str
            Path to s3 location where intermediate artifacts will be uploaded and trained models should be saved.
            This has to be provided because s3 buckets are unique globally, so it is hard to create one for you.
            If you only provided the bucket but not the subfolder, a time-stamped folder called "YOUR_BUCKET/ag-[TIMESTAMP]" will be created.
            If you provided both the bucket and the subfolder, then we will use that instead.
            Note: To call `fit()` twice and save all results of each fit,
            you must either specify different `cloud_output_path` locations or only provide the bucket but not the subfolder.
            Otherwise files from first `fit()` will be overwritten by second `fit()`.
        local_output_path: str
            Path to directory where downloaded trained predictor, batch transform results, and intermediate outputs should be saved
            If unspecified, a time-stamped folder called "AutogluonCloudPredictor/ag-[TIMESTAMP]"
            will be created in the working directory to store all downloaded trained predictor, batch transform results, and intermediate outputs.
            Note: To call `fit()` twice and save all results of each fit,
            you must specify different `local_output_path` locations or don't specify `local_output_path` at all.
            Otherwise files from first `fit()` will be overwritten by second `fit()`.
        backend: str, default = "sagemaker"
            The backend to use. Valid options are: "sagemaker" and "ray_aws".
            SageMaker backend supports training, deploying and batch inference on AWS SageMaker. Only single instance training is supported.
            RayAWS backend supports distributed training by creating an ephemeral ray cluster on AWS. Deployment and batch inferenc are not supported yet.
        verbosity : int, default = 2
            Verbosity levels range from 0 to 4 and control how much information is printed.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
            If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
            where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels).
        """
        self.verbosity = verbosity
        cloud_logger = logging.getLogger("autogluon.cloud")
        set_logger_verbosity(self.verbosity, logger=cloud_logger)
        self.local_output_path = self._setup_local_output_path(local_output_path)
        self.cloud_output_path = self._setup_cloud_output_path(cloud_output_path)
        self.backend: Backend = BackendFactory.get_backend(
            backend=self.backend_map[backend],
            local_output_path=self.local_output_path,
            cloud_output_path=self.cloud_output_path,
            predictor_type=self.predictor_type,
        )

    @property
    @abstractmethod
    def predictor_type(self) -> str:
        """
        Type of the underneath AutoGluon Predictor
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def backend_map(self) -> Dict:
        """
        Map between general backend to module specific backend
        """
        raise NotImplementedError

    @property
    def is_fit(self) -> bool:
        """
        Whether this CloudPredictor is fitted already
        """
        return self.backend.is_fit

    @property
    def endpoint_name(self) -> Optional[str]:
        """
        Return the CloudPredictor deployed endpoint name
        """
        if self.backend.endpoint:
            return self.backend.endpoint.endpoint_name
        return None

    def generate_default_permission(self, **kwargs) -> Dict[str, str]:
        """
        Generate required permission file in json format for CloudPredictor with your choice of backend.
        Users can use the generated files to create an entity for themselves.
        IMPORTANT: Make sure you review both files before creating the entity!

        Parameters
        ----------
        kwargs:
            Refer to the `generate_default_permission` of the specified backend for a list of parameters

        Return
        ------
        A dict containing the trust relationship and IAM policy files paths
        """
        return self.backend.generate_default_permission(**kwargs)

    def info(self) -> Dict[str, Any]:
        """
        Return general info about CloudPredictor
        """
        info = dict(
            local_output_path=self.local_output_path,
            cloud_output_path=self.cloud_output_path,
            fit_job=self.backend.get_fit_job_info(),
            recent_batch_inference_job=self.backend.get_batch_inference_job_info(),
            batch_inference_jobs=self.backend.get_batch_inference_jobs(),
            endpoint=self.endpoint_name,
        )
        return info

    def _setup_local_output_path(self, path):
        if path is None:
            utcnow = datetime.utcnow()
            timestamp = utcnow.strftime("%Y%m%d_%H%M%S")
            path = f"AutogluonCloudPredictor{os.path.sep}ag-{timestamp}{os.path.sep}"
        path = setup_outputdir(path)
        util_path = os.path.join(path, "utils")
        try:
            os.makedirs(util_path)
        except FileExistsError:
            logger.warning(
                f"Warning: path already exists! This predictor may overwrite an existing predictor! path='{path!r}'"
            )
        return os.path.abspath(path)

    def _setup_cloud_output_path(self, path):
        if path.endswith("/"):
            path = path[:-1]
        path_cleaned = path
        try:
            path_cleaned = path.split("://", 1)[1]
        except Exception:
            pass
        path_split = path_cleaned.split("/", 1)
        # If user only provided the bucket, we create a subfolder with timestamp for them
        if len(path_split) == 1:
            path = os.path.join(path, f"ag-{sagemaker.utils.sagemaker_timestamp()}")
        if is_s3_url(path):
            return path
        return "s3://" + path

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
        timeout: int = 24 * 60 * 60,
        wait: bool = True,
        backend_kwargs: Optional[Dict] = None,
    ) -> CloudPredictor:
        """
        Fit the predictor with the backend.

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
            If not specified, will decide by the backend
        volumes_size: int, default = 256
            Size in GB of the EBS volume to use for storing input data during training (default: 256).
            Must be large enough to store training data if File Mode is used (which is the default).
        timeout: int, default = 24*60*60
            Timeout in seconds for training. This timeout doesn't include time for pre-processing or launching up the training job.
        wait: bool, default = True
            Whether the call should wait until the job completes
            To be noticed, the function won't return immediately because there are some preparations needed prior fit.
            Use `get_fit_job_status` to get job status.
        backend_kwargs: dict, default = None
            Any extra arguments needed to pass to the underneath backend.
            For SageMaker backend, valid keys are:
                1. autogluon_sagemaker_estimator_kwargs
                    Any extra arguments needed to initialize AutoGluonSagemakerEstimator
                    Please refer to https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Estimator for all options
                2. fit_kwargs
                    Any extra arguments needed to pass to fit.
                    Please refer to https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Estimator.fit for all options
            For RayAWS backend, valid keys are:
                1. custom_config: Optional[Union[str, Dict[str, Any]]] = None,
                    The custom cluster configuration.
                    Please refer to https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-configuration.html#cluster-yaml-configuration-options for details
                2. cluster_name: Optional[str] = None,
                    The name of the ephemeral cluster being created.
                    If not specified, will be auto-generated with format f"ag_ray_aws_default_{timestamp}"
                3. initialization_commands: Optional[List[str]], default = None
                    The initialization commands of the ray cluster.
                    If not specified, will contain a default ECR login command to be able to pull AG DLC image, i.e.
                        - aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
                    To learn more about initialization_commands,
                        https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-configuration.html#initialization-commands
        Returns
        -------
        `CloudPredictor` object. Returns self.
        """  # noqa: E501
        assert (
            not self.backend.is_fit
        ), "Predictor is already fit! To fit additional models, create a new `CloudPredictor`"
        if backend_kwargs is None:
            backend_kwargs = {}
        backend_kwargs = self.backend.parse_backend_fit_kwargs(backend_kwargs)
        self.backend.fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            image_column=image_column,
            leaderboard=leaderboard,
            framework_version=framework_version,
            job_name=job_name,
            instance_type=instance_type,
            instance_count=instance_count,
            volume_size=volume_size,
            custom_image_uri=custom_image_uri,
            timeout=timeout,
            wait=wait,
            **backend_kwargs,
        )

        return self

    def attach_job(self, job_name: str) -> None:
        """
        Attach to a sagemaker training job.
        This is useful when the local process crashed and you want to reattach to the previous job

        Parameters
        ----------
        job_name: str
            The name of the job being attached
        """
        self.backend.attach_job(job_name)

    def get_fit_job_status(self) -> str:
        """
        Get the status of the training job.
        This is useful when the user made an asynchronous call to the `fit()` function

        Returns
        -------
        str,
        Valid Values: InProgress | Completed | Failed | Stopping | Stopped | NotCreated
        """
        return self.backend.get_fit_job_status()

    def get_fit_job_output_path(self) -> str:
        """
        Get the output path in the cloud of the trained artifact

        Returns
        -------
        str,
            Output path of the job
        """
        return self.backend.get_fit_job_output_path()

    def download_trained_predictor(self, save_path: Optional[str] = None) -> str:
        """
        Download the trained predictor from the cloud.

        Parameters
        ----------
        save_path: str
            Path to save the model.
            If None, CloudPredictor will create a folder 'AutogluonModels' for the model under `local_output_path`.

        Returns
        -------
        save_path: str
            Path to the saved model.
        """
        path = self.backend.get_fit_job_output_path()
        assert (
            path is not None
        ), "No fit job associated with this CloudPredictor. Either attach to a fit job with `attach_job()` or start one with `fit()`"
        if not save_path:
            save_path = self.local_output_path
        save_path = self._download_predictor(path, save_path)
        return save_path

    def _get_local_predictor_cls(self):
        raise NotImplementedError

    def to_local_predictor(self, save_path: Optional[str] = None, **kwargs):
        """
        Convert the Cloud trained predictor to a local AutoGluon Predictor.

        Parameters
        ----------
        save_path: str
            Path to save the model.
            If None, CloudPredictor will create a folder for the model.
        kwargs:
            Additional args to be passed to `load` call of the underneath predictor

        Returns
        -------
        AutoGluon Predictor,
            TabularPredictor or MultiModalPredictor based on `predictor_type`
        """
        predictor_cls = self._get_local_predictor_cls()
        local_model_path = self.download_trained_predictor(save_path)
        return predictor_cls.load(local_model_path, **kwargs)

    def deploy(
        self,
        predictor_path: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        framework_version: str = "latest",
        instance_type: str = "ml.m5.2xlarge",
        initial_instance_count: int = 1,
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        backend_kwargs: Optional[Dict] = None,
    ) -> None:
        """
        Deploy a predictor to an endpoint, which can be used to do real-time inference later.

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
        backend_kwargs: dict, default = None
            Any extra arguments needed to pass to the underneath backend.
            For SageMaker backend, valid keys are:
                1. model_kwargs: dict, default = dict()
                    Any extra arguments needed to initialize Sagemaker Model
                    Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#model for all options
                2. deploy_kwargs
                    Any extra arguments needed to pass to deploy.
                    Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#sagemaker.model.Model.deploy for all options
        """
        if backend_kwargs is None:
            backend_kwargs = {}
        backend_kwargs = self.backend.parse_backend_deploy_kwargs(backend_kwargs)
        self.backend.deploy(
            predictor_path=predictor_path,
            endpoint_name=endpoint_name,
            framework_version=framework_version,
            instance_type=instance_type,
            initial_instance_count=initial_instance_count,
            custom_image_uri=custom_image_uri,
            wait=wait,
            **backend_kwargs,
        )

    def attach_endpoint(self, endpoint: Union[str, Endpoint]) -> None:
        """
        Attach the current CloudPredictor to an existing endpoint.

        Parameters
        ----------
        endpoint: str or  :class:`Endpoint`
            If str is passed, it should be the name of the endpoint being attached to.
        """
        self.backend.attach_endpoint(endpoint)

    def detach_endpoint(self) -> Endpoint:
        """
        Detach the current endpoint and return it.

        Returns
        -------
        `Endpoint` object.
        """
        return self.backend.detach_endpoint()

    def predict_real_time(
        self,
        test_data: Union[str, pd.DataFrame],
        test_data_image_column: Optional[str] = None,
        accept: str = "application/x-parquet",
    ) -> pd.Series:
        """
        Predict with the deployed endpoint. A deployed endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict()` instead.

        Parameters
        ----------
        test_data: Union(str, pandas.DataFrame)
            The test data to be inferenced. Can be a pandas.DataFrame, or a local path to csv file.
        test_data_image_column: default = None
            If provided a csv file or pandas.DataFrame as the test_data and test_data involves image modality,
            you must specify the column name corresponding to image paths.
            The path MUST be an abspath
        accept: str, default = application/x-parquet
            Type of accept output content.
            Valid options are application/x-parquet, text/csv, application/json

        Returns
        -------
        Pandas.Series
        Predict results in Series
        """
        return self.backend.predict_real_time(
            test_data=test_data, test_data_image_column=test_data_image_column, accept=accept
        )

    def predict_proba_real_time(
        self,
        test_data: Union[str, pd.DataFrame],
        test_data_image_column: Optional[str] = None,
        accept: str = "application/x-parquet",
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Predict probability with the deployed endpoint. A deployed endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict_proba()` instead.
        If your problem_type is regression, this functions identically to `predict_real_time`, returning the same output.

        Parameters
        ----------
        test_data: Union(str, pandas.DataFrame)
            The test data to be inferenced. Can be a pandas.DataFrame, or a local path to csv file.
        test_data_image_column: default = None
            If provided a csv file or pandas.DataFrame as the test_data and test_data involves image modality,
            you must specify the column name corresponding to image paths.
            The path MUST be an abspath
        accept: str, default = application/x-parquet
            Type of accept output content.
            Valid options are application/x-parquet, text/csv, application/json

        Returns
        -------
        Pandas.DataFrame or Pandas.Series
            Will return a Pandas.Series when it's a regression problem. Will return a Pandas.DataFrame otherwise
        """
        return self.backend.predict_proba_real_time(
            test_data=test_data, test_data_image_column=test_data_image_column, accept=accept
        )

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
        backend_kwargs: Optional[Dict] = None,
    ) -> Optional[pd.Series]:
        """
        Batch inference.
        When minimizing latency isn't a concern, then the batch transform functionality may be easier, more scalable, and more appropriate.
        If you want to minimize latency, use `predict_real_time()` instead.

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
        backend_kwargs: dict, default = None
            Any extra arguments needed to pass to the underneath backend.
            For SageMaker backend, valid keys are:
                1. download: bool, default = True
                    Whether to download the batch transform results to the disk and load it after the batch transform finishes.
                    Will be ignored if `wait` is `False`.
                2. persist: bool, default = True
                    Whether to persist the downloaded batch transform results on the disk.
                    Will be ignored if `download` is `False`
                3. save_path: str, default = None,
                    Path to save the downloaded result.
                    Will be ignored if `download` is `False`.
                    If None, CloudPredictor will create one.
                    If `persist` is `False`, file would first be downloaded to this path and then removed.
                4. model_kwargs: dict, default = dict()
                    Any extra arguments needed to initialize Sagemaker Model
                    Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#model for all options
                5. transformer_kwargs: dict
                    Any extra arguments needed to pass to transformer.
                    Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html#sagemaker.transformer.Transformer for all options.
                6. transform_kwargs:
                    Any extra arguments needed to pass to transform.
                    Please refer to
                    https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html#sagemaker.transformer.Transformer.transform for all options.

        Returns
        -------
        Optional Pandas.Series
            Predict results in Series if `download` is True
            None if `download` is False
        """
        if backend_kwargs is None:
            backend_kwargs = {}
        backend_kwargs = self.backend.parse_backend_predict_kwargs(backend_kwargs)
        return self.backend.predict(
            test_data=test_data,
            test_data_image_column=test_data_image_column,
            predictor_path=predictor_path,
            framework_version=framework_version,
            job_name=job_name,
            instance_type=instance_type,
            instance_count=instance_count,
            custom_image_uri=custom_image_uri,
            wait=wait,
            **backend_kwargs,
        )

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
        backend_kwargs: Optional[Dict] = None,
    ) -> Optional[Union[Tuple[pd.Series, Union[pd.DataFrame, pd.Series]], Union[pd.DataFrame, pd.Series]]]:
        """
        Batch inference
        When minimizing latency isn't a concern, then the batch transform functionality may be easier, more scalable, and more appropriate.
        If you want to minimize latency, use `predict_real_time()` instead.

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
        backend_kwargs: dict, default = None
            Any extra arguments needed to pass to the underneath backend.
            For SageMaker backend, valid keys are:
                1. download: bool, default = True
                    Whether to download the batch transform results to the disk and load it after the batch transform finishes.
                    Will be ignored if `wait` is `False`.
                2. persist: bool, default = True
                    Whether to persist the downloaded batch transform results on the disk.
                    Will be ignored if `download` is `False`
                3. save_path: str, default = None,
                    Path to save the downloaded result.
                    Will be ignored if `download` is `False`.
                    If None, CloudPredictor will create one.
                    If `persist` is `False`, file would first be downloaded to this path and then removed.
                4. model_kwargs: dict, default = dict()
                    Any extra arguments needed to initialize Sagemaker Model
                    Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#model for all options
                5. transformer_kwargs: dict
                    Any extra arguments needed to pass to transformer.
                    Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html#sagemaker.transformer.Transformer for all options.
                6. transform_kwargs:
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
        if backend_kwargs is None:
            backend_kwargs = {}
        backend_kwargs = self.backend.parse_backend_predict_kwargs(backend_kwargs)
        return self.backend.predict_proba(
            test_data=test_data,
            test_data_image_column=test_data_image_column,
            include_predict=include_predict,
            predictor_path=predictor_path,
            framework_version=framework_version,
            job_name=job_name,
            instance_type=instance_type,
            instance_count=instance_count,
            custom_image_uri=custom_image_uri,
            wait=wait,
            **backend_kwargs,
        )

    def get_batch_inference_job_info(self, job_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get general info of the batch inference job.
        If job_name not specified, return the info of the most recent batch inference job
        """
        return self.backend.get_batch_inference_job_info(job_name)

    def get_batch_inference_job_status(self, job_name: Optional[str] = None) -> str:
        """
        Get the status of the batch inference job.
        This is useful when the user made an asynchronous call to the `predict()` function

        Parameters
        ----------
        job_name: str
            The name of the job being checked.
            If None, will check the most recent job status.

        Returns
        -------
        str,
        Valid Values: InProgress | Completed | Failed | Stopping | Stopped | NotCreated
        """
        return self.backend.get_batch_inference_job_status(job_name)

    def cleanup_deployment(self) -> None:
        """
        Delete the deployed endpoint and other artifacts
        """
        self.backend.cleanup_deployment()

    def _download_predictor(self, path, save_path):
        logger.log(20, "Downloading trained models to local directory")
        predictor_bucket, predictor_key_prefix = s3_path_to_bucket_prefix(path)
        tarball_path = os.path.join(save_path, "model.tar.gz")
        s3 = boto3.client("s3")
        s3.download_file(predictor_bucket, predictor_key_prefix, tarball_path)
        logger.log(20, "Extracting the trained model tarball")
        save_path = os.path.join(save_path, "AutoGluonModels")
        unzip_file(tarball_path, save_path)
        return save_path

    def save(self, silent: bool = False) -> None:
        """
        Save the CloudPredictor so that user can later reload the predictor to gain access to deployed endpoint.
        """
        path = self.local_output_path
        predictor_file_name = self.predictor_file_name
        save_pkl.save(path=os.path.join(path, predictor_file_name), object=self)

        if not silent:
            logger.log(
                20,
                f"{type(self).__name__} saved. To load, use: predictor = {type(self).__name__}.load('{self.local_output_path!r}')",
            )

    @classmethod
    def load(cls, path: str, verbosity: Optional[int] = None) -> CloudPredictor:
        """
        Load the CloudPredictor

        Parameters
        ----------
        path: str
            The path to directory in which this Predictor was previously saved

        Returns
        -------
        `CloudPredictor` object.
        """
        if verbosity is not None:
            set_logger_verbosity(verbosity, logger=logger)  # Reset logging after load (may be in new Python session)
        if path is None:
            raise ValueError("path cannot be None in load()")

        path = setup_outputdir(path, warn_if_exist=False)  # replace ~ with absolute path if it exists
        predictor: CloudPredictor = load_pkl.load(path=os.path.join(path, cls.predictor_file_name))
        # TODO: Version compatibility check
        return predictor
