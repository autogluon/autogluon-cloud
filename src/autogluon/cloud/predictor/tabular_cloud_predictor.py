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

    def _load_predict_real_time_test_data(self, test_data, test_data_image_column):
        if isinstance(test_data, str):
            test_data = load_pd.load(test_data)
        if isinstance(test_data, pd.DataFrame):
            if test_data_image_column is not None:
                test_data = convert_image_path_to_encoded_bytes_in_dataframe(test_data, test_data_image_column)

        return test_data

    def predict_real_time(
        self,
        test_data: Union[str, pd.DataFrame],
        test_data_image_column: Optional[str] = None,
        accept: str = "application/x-parquet",
    ):
        """
        Predict with the deployed SageMaker endpoint. A deployed SageMaker endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict()` instead.

        Parameters
        ----------
        test_data: Union(str, pandas.DataFrame)
            The test data to be inferenced.
            Can be a pandas.DataFrame or a local path to a csv file.
            When predicting multimodality with image modality:
                You need to specify `test_data_image_column`, and make sure the image column contains relative path to the image.
        test_data_image_column: default = None
            If test_data involves image modality, you must specify the column name corresponding to image paths.
            The path MUST be an abspath
        accept: str, default = application/x-parquet
            Type of accept output content.
            Valid options are application/x-parquet, text/csv, application/json

        Returns
        -------
        Pandas.Series
        Predict results in Series
        """
        self._validate_predict_real_time_args(accept)
        test_data = self._load_predict_real_time_test_data(
            test_data=test_data, test_data_image_column=test_data_image_column
        )
        pred, _ = self._predict_real_time(test_data=test_data, accept=accept)

        return pred

    def predict_proba_real_time(
        self,
        test_data: Union[str, pd.DataFrame],
        test_data_image_column: Optional[str] = None,
        accept: str = "application/x-parquet",
    ):
        """
        Predict with the deployed SageMaker endpoint. A deployed SageMaker endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict()` instead.

        Parameters
        ----------
        test_data: Union(str, pandas.DataFrame)
            The test data to be inferenced.
            Can be a pandas.DataFrame or a local path to a csv file.
            When predicting multimodality with image modality:
                You need to specify `test_data_image_column`, and make sure the image column contains relative path to the image.
        test_data_image_column: default = None
            If test_data involves image modality, you must specify the column name corresponding to image paths.
            The path MUST be an abspath
        accept: str, default = application/x-parquet
            Type of accept output content.
            Valid options are application/x-parquet, text/csv, application/json

        Returns
        -------
        Pandas.DataFrame or Pandas.Series
            Will return a Pandas.Series when it's a regression problem. Will return a Pandas.DataFrame otherwise
        """
        self._validate_predict_real_time_args(accept)
        test_data = self._load_predict_real_time_test_data(
            test_data=test_data, test_data_image_column=test_data_image_column
        )
        pred, proba = self._predict_real_time(test_data=test_data, accept=accept)

        if proba is None:
            return pred

        return proba
