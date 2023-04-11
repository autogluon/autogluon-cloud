import logging
from typing import Dict

from ..backend.constant import RAY_AWS, SAGEMAKER, TABULAR_RAY_AWS, TABULAR_SAGEMAKER
from .cloud_predictor import CloudPredictor

logger = logging.getLogger(__name__)


class TabularCloudPredictor(CloudPredictor):
    predictor_file_name = "TabularCloudPredictor.pkl"

    @property
    def predictor_type(self):
        """
        Type of the underneath AutoGluon Predictor
        """
        return "tabular"

    @property
    def backend_map(self) -> Dict:
        """
        Map between general backend to module specific backend
        """
        return {SAGEMAKER: TABULAR_SAGEMAKER, RAY_AWS: TABULAR_RAY_AWS}

    def _get_local_predictor_cls(self):
        from autogluon.tabular import TabularPredictor

        predictor_cls = TabularPredictor
        return predictor_cls
