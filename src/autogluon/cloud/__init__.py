import logging

from autogluon.common.utils.log_utils import _add_stream_handler

from .cloud_setup import bootstrap, register, status, teardown
from .model.foundation_model import TimeSeriesFoundationModel
from .predictor import MultiModalCloudPredictor, TabularCloudPredictor, TimeSeriesCloudPredictor

_add_stream_handler()
logging.getLogger(__name__).setLevel(logging.INFO)

__all__ = [
    "MultiModalCloudPredictor",
    "TabularCloudPredictor",
    "TimeSeriesCloudPredictor",
    "TimeSeriesFoundationModel",
    "bootstrap",
    "register",
    "status",
    "teardown",
]
