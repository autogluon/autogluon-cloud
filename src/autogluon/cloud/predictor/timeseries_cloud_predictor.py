from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

from ..backend.constant import SAGEMAKER, TIMESERIES_SAGEMAKER
from .cloud_predictor import CloudPredictor

logger = logging.getLogger(__name__)


class TimeSeriesCloudPredictor(CloudPredictor):
    predictor_file_name = "TimeSeriesCloudPredictor.pkl"

    @property
    def predictor_type(self):
        """
        Type of the underneath AutoGluon Predictor
        """
        return "timeseries"

    @property
    def backend_map(self) -> Dict:
        """
        Map between general backend to module specific backend
        """
        return {SAGEMAKER: TIMESERIES_SAGEMAKER}

    def _get_local_predictor_cls(self):
        from autogluon.timeseries import TimeSeriesPredictor

        predictor_cls = TimeSeriesPredictor
        return predictor_cls

    def predict_proba_real_time(self, **kwargs) -> pd.DataFrame:
        raise ValueError(f"{self.__class__.__name__} does not support predict_proba operation.")

    def predict_proba(
        self,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        raise ValueError(f"{self.__class__.__name__} does not support predict_proba operation.")
