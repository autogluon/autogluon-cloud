import logging

from ..backend.constant import RAY_AWS, SAGEMAKER, TABULAR_RAY_AWS, TABULAR_SAGEMAKER
from .cloud_predictor import CloudPredictor

logger = logging.getLogger(__name__)


class TabularCloudPredictor(CloudPredictor):
    """Train and deploy AutoGluon tabular models (classification and regression) on AWS SageMaker.

    Wraps :class:`autogluon.tabular.TabularPredictor` (`docs <https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html>`_)
    and runs ``fit``, ``predict``, and endpoint deployment as managed SageMaker jobs.
    """

    predictor_file_name = "TabularCloudPredictor.pkl"
    backend_map = {SAGEMAKER: TABULAR_SAGEMAKER, RAY_AWS: TABULAR_RAY_AWS}

    @property
    def predictor_type(self):
        """
        Type of the underneath AutoGluon Predictor
        """
        return "tabular"

    def _get_local_predictor_cls(self):
        from autogluon.tabular import TabularPredictor

        predictor_cls = TabularPredictor
        return predictor_cls
