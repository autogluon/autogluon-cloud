import copy
import logging
import os
from typing import Optional, Union

import pandas as pd
import yaml

from autogluon.common.loaders import load_pd

from ..utils.utils import convert_image_path_to_encoded_bytes_in_dataframe
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

    def _get_local_predictor_cls(self):
        from autogluon.tabular import TabularPredictor

        predictor_cls = TabularPredictor
        return predictor_cls
