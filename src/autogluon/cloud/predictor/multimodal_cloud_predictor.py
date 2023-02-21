import logging

from .cloud_predictor import CloudPredictor

logger = logging.getLogger(__name__)


class MultiModalCloudPredictor(CloudPredictor):
    predictor_file_name = "MultiModalCloudPredictor.pkl"

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
