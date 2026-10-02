import logging

from autogluon.common.utils.log_utils import _add_stream_handler

from .cloud_setup import bootstrap, register, status, teardown
from .config import SageMakerConfig
from .endpoint.tabular_endpoint import TabularEndpoint
from .endpoint.timeseries_endpoint import TimeSeriesEndpoint
from .model.foundation_model import FoundationModel, TabularFoundationModel, TimeSeriesFoundationModel
from .predictor import MultiModalCloudPredictor, TabularCloudPredictor, TimeSeriesCloudPredictor

_add_stream_handler()
logging.getLogger(__name__).setLevel(logging.INFO)

__all__ = [
    "FoundationModel",
    "MultiModalCloudPredictor",
    "TabularCloudPredictor",
    "TabularEndpoint",
    "TabularFoundationModel",
    "SageMakerConfig",
    "TimeSeriesCloudPredictor",
    "TimeSeriesEndpoint",
    "TimeSeriesFoundationModel",
    "bootstrap",
    "register",
    "status",
    "teardown",
]
