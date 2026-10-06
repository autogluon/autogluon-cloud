import logging
from abc import abstractmethod
from typing import Any, Dict, Optional, Union

from ..utils.aws_utils import setup_sagemaker_session
from ..utils.constants import MODEL_ARTIFACT_NAME
from ..utils.dlc_utils import infer_framework_version_from_image_uri
from ..utils.job_logs import TRAINING_JOB_LOG_GROUP, TRANSFORM_JOB_LOG_GROUP, wait_for_job
from ..utils.sagemaker_api import delete_quietly
from .remote_job import RemoteJob

logger = logging.getLogger(__name__)


class SageMakerJob(RemoteJob):
    _LOG_GROUP: str

    def __init__(self, session=None):
        self.session = session or setup_sagemaker_session()
        self._job_name = None
        self._output_filename = ""

    @classmethod
    @abstractmethod
    def attach(cls, job_name, session=None):
        """
        Reattach to a job given its name.

        Parameters:
        -----------
        job_name: str
            Name of the job to be attached.
        """
        raise NotImplementedError

    @abstractmethod
    def info(self) -> dict:
        """
        Give general information about the job.

        Returns:
        ------
        dict
            A dictionary containing the general information about the job.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, **kwargs):
        """Execute the job"""
        raise NotImplementedError

    @abstractmethod
    def _describe(self) -> Dict[str, Any]:
        """Return the ``Describe*Job`` response for the job."""
        raise NotImplementedError

    @abstractmethod
    def _get_job_status(self):
        raise NotImplementedError

    @abstractmethod
    def _get_output_path(self):
        raise NotImplementedError

    @abstractmethod
    def _get_hyperparameters(self):
        raise NotImplementedError

    @property
    def job_name(self):
        return self._job_name

    @property
    def completed(self):
        if not self.job_name:
            return False
        return self.get_job_status() == "Completed"

    def get_job_status(self) -> Optional[str]:
        """
        Get job status

        Returns:
        --------
        str:
            Valid Values: InProgress | Completed | Failed | Stopping | Stopped | NotCreated
        """
        if not self.job_name:
            return "NotCreated"
        return self._get_job_status()

    def get_output_path(self) -> Optional[str]:
        """
        Get the output path of the job generated artifacts if any.

        Returns:
        --------
        Optional[str]:
            Output path of the job generated artifacts if any.
            If no artifact, return None
        """
        if not self.completed:
            return None
        return self._get_output_path()

    def get_hyperparameters(self) -> Dict[str, Union[int, str]]:
        """
        Get hyperparameters of the job

        Returns:
        --------
        dict:
            Hyperparameters of the training job
        """
        return self._get_hyperparameters()

    def wait(self, logs: bool = True) -> str:
        """Block until the job reaches a terminal state, streaming its CloudWatch logs if ``logs`` is True.

        Does not raise if the job fails. Returns the final status (Completed | Failed | Stopped).
        """
        assert self.job_name, "The job has not been started"
        status = wait_for_job(
            self.get_job_status,
            job_name=self.job_name,
            log_group=self._LOG_GROUP,
            logs_client=self.session.boto_session.client("logs") if logs else None,
        )
        if status != "Completed":
            logger.error(
                f"SageMaker job {self.job_name} finished with status {status}: {self._describe().get('FailureReason')}"
            )
        return status

    def _wait_until_completed(self) -> None:
        """Wait for the job with logs and raise if it does not complete successfully."""
        status = self.wait(logs=True)
        if status != "Completed":
            raise RuntimeError(
                f"SageMaker job {self.job_name} finished with status {status}: {self._describe().get('FailureReason')}"
            )

    def __getstate__(self):
        state_dict = self.__dict__.copy()
        state_dict["session"] = None
        return state_dict

    def __setstate__(self, state):
        self.__dict__ = state


class SageMakerFitJob(SageMakerJob):
    _LOG_GROUP = TRAINING_JOB_LOG_GROUP

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._framework_version = None
        self._output_filename = MODEL_ARTIFACT_NAME

    @classmethod
    def attach(cls, job_name, session=None):
        obj = cls(session=session)
        obj._job_name = job_name
        obj._wait_until_completed()
        # Jobs created from an algorithm resource (AlgorithmName) have no TrainingImage
        training_image = obj._describe()["AlgorithmSpecification"].get("TrainingImage")
        obj._framework_version = infer_framework_version_from_image_uri(training_image) if training_image else None
        return obj

    @property
    def framework_version(self):
        return self._framework_version

    def info(self):
        info = dict(
            name=self.job_name,
            status=self.get_job_status(),
            framework_version=self.framework_version,
            artifact_path=self.get_output_path(),
            hyperparameters=self.get_hyperparameters(),
        )
        return info

    def _describe(self) -> Dict[str, Any]:
        return self.session.sagemaker_client.describe_training_job(TrainingJobName=self.job_name)

    def _get_job_status(self):
        return self._describe()["TrainingJobStatus"]

    def _get_output_path(self):
        return self._describe()["ModelArtifacts"]["S3ModelArtifacts"]

    def _get_hyperparameters(self):
        if self.job_name:
            return self._describe().get("HyperParameters")
        return None

    def run(
        self,
        training_job_request: Dict[str, Any],
        framework_version: Optional[str],
        wait: bool,
    ):
        """Create the training job from a ``CreateTrainingJob`` request and optionally wait for it to finish."""
        job_name = training_job_request["TrainingJobName"]
        logger.log(20, f"Start sagemaker training job `{job_name}`")
        try:
            self.session.sagemaker_client.create_training_job(**training_job_request)
            self._job_name = job_name
            self._framework_version = framework_version
            if wait:
                self._wait_until_completed()
        except Exception as e:
            logger.error(f"Training failed. Please check sagemaker console training jobs {job_name} for details.")
            raise e


class SageMakerBatchTransformationJob(SageMakerJob):
    _LOG_GROUP = TRANSFORM_JOB_LOG_GROUP

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._output_filename = ""

    @classmethod
    def attach(cls, job_name, session=None):
        raise NotImplementedError

    def info(self):
        info = dict(
            name=self.job_name,
            status=self.get_job_status(),
            hyperparameters=self.get_hyperparameters(),
            result_path=self._get_output_path(),
        )
        return info

    def _describe(self) -> Dict[str, Any]:
        return self.session.sagemaker_client.describe_transform_job(TransformJobName=self.job_name)

    def _get_job_status(self):
        return self._describe()["TransformJobStatus"]

    def _get_output_path(self):
        return self._describe()["TransformOutput"]["S3OutputPath"] + "/" + self._output_filename

    def _delete_model(self, model_name: str) -> None:
        self.session.sagemaker_client.delete_model(ModelName=model_name)

    def run(
        self,
        transform_job_request: Dict[str, Any],
        model_name: str,
        wait: bool,
    ):
        """Create the transform job from a ``CreateTransformJob`` request.

        ``model_name`` (the model created for this job) is deleted once the job finishes (``wait=True``) or fails to
        start. With ``wait=False`` the model is kept, since the job still needs it.
        """
        job_name = transform_job_request["TransformJobName"]
        try:
            logger.log(20, "Transforming")
            self.session.sagemaker_client.create_transform_job(**transform_job_request)
            self._job_name = job_name
            if wait:
                self._wait_until_completed()
            logger.log(20, "Transform done")
        except Exception as e:
            delete_quietly(self.session.sagemaker_client.delete_model, ModelName=model_name)
            raise e

        input_uri = transform_job_request["TransformInput"]["DataSource"]["S3DataSource"]["S3Uri"]
        self._output_filename = input_uri.split("/")[-1] + ".out"

        if wait:
            self._delete_model(model_name)
            logger.log(20, f"Predict results have been saved to {self.get_output_path()}")
        else:
            logger.log(
                20,
                "Predict asynchronously. You can use `info()` or `get_job_status()` to check the status.",
            )

    def _get_hyperparameters(self):
        """
        Get hyperparameters of the batch transformation job
        Currently batch transformation jobs don't have hyperparameters
        """
        return {}
