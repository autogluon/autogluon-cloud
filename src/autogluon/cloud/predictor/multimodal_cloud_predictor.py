import logging

from ..backend.constant import MULTIMODL_SAGEMAKER, SAGEMAKER
from .cloud_predictor import CloudPredictor

logger = logging.getLogger(__name__)


class MultiModalCloudPredictor(CloudPredictor):
    """Train and deploy AutoGluon multimodal models (image, text, tabular) on AWS SageMaker.

    Wraps :class:`autogluon.multimodal.MultiModalPredictor` (`docs <https://auto.gluon.ai/stable/api/autogluon.multimodal.MultiModalPredictor.html>`_)
    and runs ``fit``, ``predict``, and endpoint deployment as managed SageMaker jobs.
    """

    predictor_file_name = "MultiModalCloudPredictor.pkl"
    backend_map = {SAGEMAKER: MULTIMODL_SAGEMAKER}

    @property
    def predictor_type(self) -> str:
        """
        Type of the underneath AutoGluon Predictor
        """
        return "multimodal"

    def _get_local_predictor_cls(self):
        from autogluon.multimodal import MultiModalPredictor

        predictor_cls = MultiModalPredictor
        return predictor_cls
