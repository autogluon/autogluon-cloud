import logging
from typing import Any

import boto3

from ..utils.aws_utils import setup_sagemaker_session
from ..utils.sagemaker_api import delete_endpoint

logger = logging.getLogger(__name__)


class Endpoint:
    """Base class for handles to SageMaker endpoints deployed through AutoGluon-Cloud."""

    def __init__(self, endpoint_name: str, session: boto3.Session | None = None):
        """
        Parameters
        ----------
        endpoint_name: str
            Name of an existing SageMaker endpoint deployed through AutoGluon-Cloud, either with ``deploy()`` of a
            cloud predictor (e.g. :meth:`autogluon.cloud.TabularCloudPredictor.deploy`) or of a foundation model
            (e.g. :meth:`autogluon.cloud.TabularFoundationModel.deploy`).
        session: boto3.Session | None, default = None
            ``boto3.Session`` used to invoke and delete the endpoint. If ``None``, the default ambient session is used.
        """
        self._endpoint_name = endpoint_name
        self._session = setup_sagemaker_session(boto_session=session)

    @property
    def endpoint_name(self) -> str:
        return self._endpoint_name

    @staticmethod
    def _pop_as_pandas(inference_kwargs: dict[str, Any]) -> None:
        # The serve scripts always pass as_pandas=True, so forwarding it would raise a duplicate-keyword TypeError.
        if inference_kwargs.pop("as_pandas", True) is not True:
            logger.warning("as_pandas must be True for real-time prediction; ignoring it.")

    def delete_endpoint(self) -> None:
        """Delete the endpoint and its backing model + endpoint config.

        SageMaker API
        -------------
        * :sm-api:`DeleteEndpoint`, :sm-api:`DeleteEndpointConfig` and :sm-api:`DeleteModel`: delete the endpoint and
          the endpoint config and model created with it.
        """
        delete_endpoint(self._endpoint_name, self._session)
