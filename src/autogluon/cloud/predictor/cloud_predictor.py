from __future__ import annotations

import copy
import logging
import os
import tarfile
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import sagemaker

from autogluon.common.loaders import load_pd, load_pkl
from autogluon.common.savers import save_pkl
from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.common.utils.s3_utils import is_s3_url, s3_path_to_bucket_prefix
from autogluon.common.utils.utils import setup_outputdir

from ..backend.backend import Backend
from ..backend.backend_factory import BackendFactory
from ..backend.constant import SAGEMAKER
from ..endpoint.endpoint import Endpoint
from ..job import SageMakerBatchTransformationJob
from ..scripts import ScriptManager
from ..utils.ag_sagemaker import AutoGluonBatchPredictor, AutoGluonRealtimePredictor
from ..utils.aws_utils import setup_sagemaker_session
from ..utils.constants import SAGEMAKER_RESOURCE_PREFIX
from ..utils.misc import MostRecentInsertedOrderedDict
from ..utils.utils import (
    convert_image_path_to_encoded_bytes_in_dataframe,
    is_image_file,
    split_pred_and_pred_proba,
    unzip_file,
)

logger = logging.getLogger(__name__)


class CloudPredictor(ABC):
    predictor_file_name = "CloudPredictor.pkl"

    def __init__(
        self, cloud_output_path: str, local_output_path: Optional[str] = None, backend=SAGEMAKER, verbosity: int = 2
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
        verbosity : int, default = 2
            Verbosity levels range from 0 to 4 and control how much information is printed.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
            If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
            where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels).
        """
        self.verbosity = verbosity
        set_logger_verbosity(self.verbosity, logger=logger)
        self.local_output_path = self._setup_local_output_path(local_output_path)
        self.cloud_output_path = self._setup_cloud_output_path(cloud_output_path)
        self.backend: Backend = BackendFactory.get_backend(self.backend_map[backend])
        self._batch_transform_jobs = MostRecentInsertedOrderedDict()

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
        return self._fit_job.completed

    @property
    def endpoint_name(self) -> str:
        """
        Return the CloudPredictor deployed endpoint name
        """
        if self.endpoint:
            return self.endpoint.endpoint_name
        return None

    @staticmethod
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
            fit_job=self._fit_job.info(),
            recent_transform_job=self._batch_transform_jobs.last_value.info()
            if len(self._batch_transform_jobs) > 0
            else None,
            transform_jobs=[job_name for job_name in self._batch_transform_jobs.keys()],
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
        instance_count: int = 1,
        volume_size: int = 100,
        custom_image_uri: Optional[str] = None,
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
        volumes_size: int, default = 30
            Size in GB of the EBS volume to use for storing input data during training (default: 30).
            Must be large enough to store training data if File Mode is used (which is the default).
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

        Returns
        -------
        `CloudPredictor` object. Returns self.
        """
        assert (
            not self.backend._fit_job.completed
        ), "Predictor is already fit! To fit additional models, create a new `CloudPredictor`"
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
            wait=wait,
            *backend_kwargs,
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
        path = self._fit_job.get_output_path()
        if not save_path:
            save_path = self.local_output_path
        save_path = self._download_predictor(path, save_path)
        return save_path

    def _get_local_predictor_cls(self):
        raise NotImplementedError

    def to_local_predictor(self, save_path: Optional[str] = None, **kwargs):
        """
        Convert the SageMaker trained predictor to a local AutoGluon Predictor.

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
        backend_kwargs = self.backend.parse_backend_deploy_kwargs(backend_kwargs)
        self.backend.prepare_deploy(realtime_predictor_cls=self._realtime_predictor_cls)
        self.backend.deploy(
            predictor_path=predictor_path,
            endpoint_name=endpoint_name,
            framework_version=framework_version,
            instance_type=instance_type,
            initial_instance_count=initial_instance_count,
            custom_image_uri=custom_image_uri,
            wait=wait,
            *backend_kwargs,
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

    def detach_endpoint(self) -> AutoGluonRealtimePredictor:
        """
        Detach the current endpoint and return it.

        Returns
        -------
        `AutoGluonRealtimePredictor` object.
        """
        assert self.endpoint is not None, "There is no attached endpoint"
        detached_endpoint = self.endpoint
        self.endpoint = None
        return detached_endpoint

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
        return self.backend.predict_realtime(
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
        return self.backend.predict_proba_realtime(
            test_data=test_data, test_data_image_column=test_data_image_column, accept=accept
        )

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

    def _prepare_image_predict_args(self, **predict_kwargs):
        split_type = None
        content_type = "application/x-image"
        predict_kwargs = copy.deepcopy(predict_kwargs)
        transformer_kwargs = predict_kwargs.pop("transformer_kwargs", dict())
        transformer_kwargs["strategy"] = "SingleRecord"

        return {"split_type": split_type, "content_type": content_type, "transformer_kwargs": transformer_kwargs}

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
        **kwargs,
    ):
        if not predictor_path:
            predictor_path = self._fit_job.get_output_path()
            assert predictor_path, "No cloud trained model found."

        if custom_image_uri:
            framework_version, py_version = None, None
            logger.log(20, f"Predicting with custom_image_uri=={custom_image_uri}")
        else:
            framework_version, py_version = self._parse_framework_version(framework_version, "inference")
            logger.log(20, f"Predicting with framework_version=={framework_version}")

        output_path = kwargs.get("output_path", None)
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

        kwargs = copy.deepcopy(kwargs)
        content_type = kwargs.pop("content_type", None)
        if "split_type" not in kwargs:
            split_type = "Line"
        else:
            split_type = kwargs.pop("split_type")
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
            **kwargs,
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
        **kwargs,
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
        **kwargs:
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
            **kwargs,
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
        **kwargs,
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
        **kwargs:
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
            **kwargs,
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

    def get_batch_transform_job_status(self, job_name: Optional[str] = None) -> str:
        """
        Get the status of the batch transform job.
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
        if not job_name:
            job_name = self._batch_transform_jobs.last
        job = self._batch_transform_jobs.get(job_name, None)
        if job:
            return job.get_job_status()
        return "NotCreated"

    def cleanup_deployment(self) -> None:
        """
        Delete endpoint, endpoint configuration and deployed model
        """
        self._delete_endpoint_model()
        self._delete_endpoint()

    def _delete_endpoint(self, delete_endpoint_config=True):
        assert self.endpoint, "There is no endpoint deployed yet"
        logger.log(20, "Deleteing endpoint")
        self.endpoint.delete_endpoint(delete_endpoint_config=delete_endpoint_config)
        logger.log(20, "Endpoint deleted")
        self.endpoint = None

    def _delete_endpoint_model(self):
        assert self.endpoint, "There is no endpoint deployed yet"
        logger.log(20, "Deleting endpoint model")
        self.endpoint.delete_model()
        logger.log(20, "Endpoint model deleted")

    def _download_predictor(self, path, save_path):
        logger.log(20, "Downloading trained models to local directory")
        predictor_bucket, predictor_key_prefix = s3_path_to_bucket_prefix(path)
        self.sagemaker_session.download_data(
            path=save_path,
            bucket=predictor_bucket,
            key_prefix=predictor_key_prefix,
        )
        logger.log(20, "Extracting the trained model tarball")
        tarball_path = os.path.join(save_path, "model.tar.gz")
        save_path = os.path.join(save_path, "AutoGluonModels")
        unzip_file(tarball_path, save_path)
        return save_path

    def save(self, silent: bool = False) -> None:
        """
        Save the CloudPredictor so that user can later reload the predictor to gain access to deployed endpoint.
        """
        path = self.local_output_path
        predictor_file_name = self.predictor_file_name
        temp_session = self.sagemaker_session
        temp_region = self._region
        self.sagemaker_session = None
        self._region = None
        temp_endpoint = None
        if self.endpoint:
            temp_endpoint = self.endpoint
            self._endpoint_saved = self.endpoint_name
            self.endpoint = None

        save_pkl.save(path=os.path.join(path, predictor_file_name), object=self)
        self.sagemaker_session = temp_session
        self._region = temp_region
        if temp_endpoint:
            self.endpoint = temp_endpoint
            self._endpoint_saved = None
        if not silent:
            logger.log(
                20,
                f"{type(self).__name__} saved. To load, use: predictor = {type(self).__name__}.load('{self.local_output_path!r}')",
            )

    def _load_jobs(self):
        self._fit_job.session = self.sagemaker_session
        for job in self._batch_transform_jobs:
            job.session = self.sagemaker_session

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
        predictor.sagemaker_session = setup_sagemaker_session()
        predictor._region = predictor.sagemaker_session.boto_region_name
        predictor._load_jobs()
        if hasattr(predictor, "_endpoint_saved") and predictor._endpoint_saved:
            predictor.attach_endpoint(predictor._endpoint_saved)
            predictor._endpoint_saved = None
        # TODO: Version compatibility check
        return predictor
