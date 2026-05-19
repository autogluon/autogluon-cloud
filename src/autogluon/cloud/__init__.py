from autogluon.common.utils.log_utils import _add_stream_handler

from .cloud_setup import bootstrap, register, status, teardown
from .predictor import MultiModalCloudPredictor, TabularCloudPredictor, TimeSeriesCloudPredictor

_add_stream_handler()

__all__ = [
    "MultiModalCloudPredictor",
    "TabularCloudPredictor",
    "TimeSeriesCloudPredictor",
    "bootstrap",
    "register",
    "status",
    "teardown",
]
