import logging

import sagemaker
from packaging.version import Version

if Version(sagemaker.__version__) >= Version("3.0"):
    raise ImportError(
        f"SageMaker SDK >= 3.0 is currently not supported (found {sagemaker.__version__}). "
        "Please downgrade: pip install -U 'sagemaker<3'"
    )

from autogluon.common.utils.log_utils import _add_stream_handler

from .cloud_setup import bootstrap, register, status, teardown
from .endpoint.timeseries_endpoint import TimeSeriesEndpoint
from .model.foundation_model import TimeSeriesFoundationModel
from .predictor import MultiModalCloudPredictor, TabularCloudPredictor, TimeSeriesCloudPredictor

_add_stream_handler()
logging.getLogger(__name__).setLevel(logging.INFO)

__all__ = [
    "MultiModalCloudPredictor",
    "TabularCloudPredictor",
    "TimeSeriesCloudPredictor",
    "TimeSeriesEndpoint",
    "TimeSeriesFoundationModel",
    "bootstrap",
    "register",
    "status",
    "teardown",
]
