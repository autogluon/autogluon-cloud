import copy
import os
from typing import Any, Dict, Optional, Union

import pandas as pd
import yaml
from sagemaker import Predictor

from autogluon.common.loaders import load_pd

from ..utils.ag_sagemaker import AutoGluonMultiModalRealtimePredictor
from ..utils.utils import convert_image_path_to_encoded_bytes_in_dataframe, is_image_file, read_image_bytes_and_encode
from .constant import TABULAR_SAGEMAKER
from .sagemaker_backend import SagemakerBackend


class TabularSagemakerBackend(SagemakerBackend):
    @property
    def name(self) -> str:
        """Name of this backend"""
        return TABULAR_SAGEMAKER

    def _construct_config(self, predictor_init_args, predictor_fit_args, leaderboard, **kwargs):
        assert self.predictor_type is not None
        if "feature_metadata" in predictor_fit_args:
            predictor_fit_args = copy.deepcopy(predictor_fit_args)
            feature_metadata = predictor_fit_args.pop("feature_metadata")
            feature_metadata = dict(
                type_map_raw=feature_metadata.type_map_raw,
                type_map_special=feature_metadata.get_type_map_special(),
            )
            assert (
                "feature_metadata" not in kwargs
            ), "feature_metadata in both `predictor_fit_args` and kwargs. This should not happen."
            kwargs["feature_metadata"] = feature_metadata
        config = dict(
            predictor_type=self.predictor_type,
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            leaderboard=leaderboard,
            **kwargs,
        )
        path = os.path.join(self.local_output_path, "utils", "config.yaml")
        with open(path, "w") as f:
            yaml.dump(config, f)
        return path
