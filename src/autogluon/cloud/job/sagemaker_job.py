import logging
from abc import abstractmethod
from typing import Any, Dict, Optional, Union

from sagemaker.core.resources import Model, TrainingJob, TransformJob
from sagemaker.core.utils.exceptions import FailedStatusError

from ..utils.aws_utils import setup_sagemaker_session
from ..utils.constants import MODEL_ARTIFACT_NAME
from ..utils.sagemaker_api import bind_core_session
from .remote_job import RemoteJob

logger = logging.getLogger(__name__)


class SageMakerJob(RemoteJob):
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
    def _describe(self):
        """Return the sagemaker-core resource describing the job."""
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
    def _boto_session(self):
        boto_session = self.session.boto_session
        bind_core_session(boto_session)
        return boto_session

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

    def wait(self, logs: bool = True) -> None:
        """Block until the job reaches a terminal state, streaming its CloudWatch logs if ``logs`` is True.

        Does not raise if the job fails; check :meth:`get_job_status` afterwards.
        """
        assert self.job_name, "The job has not been started"
        try:
            self._describe().wait(logs=logs)
        except FailedStatusError as e:
            logger.error(f"SageMaker job {self.job_name} did not complete successfully: {e}")

    def __getstate__(self):
        state_dict = self.__dict__.copy()
        state_dict["session"] = None
        return state_dict

    def __setstate__(self, state):
        self.__dict__ = state


class SageMakerFitJob(SageMakerJob):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._framework_version = None
        self._output_filename = MODEL_ARTIFACT_NAME

    @classmethod
    def attach(cls, job_name, session=None):
        # FIXME: find a way to recover framework version
        logger.warning(
            "Reattach to a job does not support real-time logging. Logs will be printed once the training job completes"
        )
        obj = cls(session=session)
        obj._job_name = job_name
        obj.wait(logs=True)
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

    def _describe(self) -> TrainingJob:
        return TrainingJob.get(
            training_job_name=self.job_name,
            session=self._boto_session,
            region=self.session.boto_region_name,
        )

    def _get_job_status(self):
        return self._describe().training_job_status

    def _get_output_path(self):
        return self._describe().model_artifacts.s3_model_artifacts

    def _get_hyperparameters(self):
        if self.job_name:
            return self._describe().hyper_parameters
        return None

    def get_input_channels(self) -> Dict[str, str]:
        """Map each input channel name of the training job to its S3 URI."""
        return {
            channel.channel_name: channel.data_source.s3_data_source.s3_uri
            for channel in self._describe().input_data_config
        }

    def run(
        self,
        training_job_request: Dict[str, Any],
        framework_version: Optional[str],
        wait: bool,
    ):
        """Create the training job from a ``TrainingJob.create`` request and optionally wait for it to finish."""
        job_name = training_job_request["training_job_name"]
        logger.log(20, f"Start sagemaker training job `{job_name}`")
        try:
            training_job = TrainingJob.create(
                **training_job_request,
                session=self._boto_session,
                region=self.session.boto_region_name,
            )
            self._job_name = job_name
            self._framework_version = framework_version
            if wait:
                training_job.wait(logs=True)
        except Exception as e:
            logger.error(f"Training failed. Please check sagemaker console training jobs {job_name} for details.")
            raise e


class SageMakerBatchTransformationJob(SageMakerJob):
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

    def _describe(self) -> TransformJob:
        return TransformJob.get(
            transform_job_name=self.job_name,
            session=self._boto_session,
            region=self.session.boto_region_name,
        )

    def _get_job_status(self):
        return self._describe().transform_job_status

    def _get_output_path(self):
        return self._describe().transform_output.s3_output_path + "/" + self._output_filename

    def _delete_model(self, model_name: str) -> None:
        bind_core_session(self.session.boto_session)
        Model(model_name=model_name).delete()

    def run(
        self,
        transform_job_request: Dict[str, Any],
        wait: bool,
    ):
        """Create the transform job from a ``TransformJob.create`` request.

        The SageMaker model referenced by the request is deleted once the job finishes (``wait=True``) or fails to
        start. With ``wait=False`` the model is kept, since the job still needs it.
        """
        job_name = transform_job_request["transform_job_name"]
        model_name = transform_job_request["model_name"]
        try:
            logger.log(20, "Transforming")
            transform_job = TransformJob.create(
                **transform_job_request,
                session=self._boto_session,
                region=self.session.boto_region_name,
            )
            self._job_name = job_name
            if wait:
                transform_job.wait(logs=True)
            logger.log(20, "Transform done")
        except Exception as e:
            self._delete_model(model_name)
            raise e

        input_uri = transform_job_request["transform_input"]["data_source"]["s3_data_source"]["s3_uri"]
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
