import logging
import os
from typing import Optional, Tuple, Union

import pandas as pd

from autogluon.common.loaders import load_pd

from ..utils.ag_sagemaker import AutoGluonMultiModalRealtimePredictor
from ..utils.utils import convert_image_path_to_encoded_bytes_in_dataframe, is_image_file, read_image_bytes_and_encode
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

    @property
    def _realtime_predictor_cls(self):
        return AutoGluonMultiModalRealtimePredictor

    def _get_local_predictor_cls(self):
        from autogluon.multimodal import MultiModalPredictor

        predictor_cls = MultiModalPredictor
        return predictor_cls

    def _load_predict_real_time_test_data(self, test_data, test_data_image_column):
        import numpy as np

        if isinstance(test_data, str):
            if is_image_file(test_data):
                test_data = [test_data]
            else:
                test_data = load_pd.load(test_data)
        if isinstance(test_data, list):
            test_data = np.array([read_image_bytes_and_encode(image) for image in test_data], dtype="object")
            content_type = "application/x-npy"
        if isinstance(test_data, pd.DataFrame):
            if test_data_image_column is not None:
                test_data = convert_image_path_to_encoded_bytes_in_dataframe(
                    dataframe=test_data, image_column=test_data_image_column
                )
            content_type = "application/x-parquet"

        return test_data, content_type

    def predict_real_time(
        self,
        test_data: Union[str, pd.DataFrame],
        test_data_image_column: Optional[str] = None,
        accept: str = "application/x-parquet",
    ) -> pd.Series:
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
            When predicting with only images:
                Can be a pandas.DataFrame or a local path to a csv file.
                    Similarly, you need to specify `test_data_image_column`, and make sure the image column contains relative path to the image.
                Or a local path to a single image file.
                Or a list of local paths to image files.
        test_data_image_column: default = None
            If provided a csv file or pandas.DataFrame as the test_data and test_data involves image modality,
            you must specify the column name corresponding to image paths.
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
        test_data, content_type = self._load_predict_real_time_test_data(
            test_data=test_data, test_data_image_column=test_data_image_column
        )
        # Providing content type here because sagemaker serializer doesn't support change content type dynamically.
        # Pass to `endpoint.predict()` call as `initial_args` instead
        pred, _ = self._predict_real_time(test_data=test_data, accept=accept, ContentType=content_type)

        return pred

    def predict_proba_real_time(
        self,
        test_data: Union[str, pd.DataFrame],
        test_data_image_column: Optional[str] = None,
        accept: str = "application/x-parquet",
    ) -> Union[pd.DataFrame, pd.Series]:
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
            When predicting with only images:
                Can be a pandas.DataFrame or a local path to a csv file.
                    Similarly, you need to specify `test_data_image_column`, and make sure the image column contains relative path to the image.
                Or a local path to a single image file.
                Or a list of local paths to image files.
        test_data_image_column: default = None
            If provided a csv file or pandas.DataFrame as the test_data and test_data involves image modality,
            you must specify the column name corresponding to image paths.
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
        test_data, content_type = self._load_predict_real_time_test_data(
            test_data=test_data, test_data_image_column=test_data_image_column
        )
        # Providing content type here because sagemaker serializer doesn't support change content type dynamically.
        # Pass to `endpoint.predict()` call as `initial_args` instead
        pred, proba = self._predict_real_time(test_data=test_data, accept=accept, ContentType=content_type)

        if proba is None:
            return pred

        return proba

    def _check_image_modality_only(self, test_data):
        image_modality_only = False
        if isinstance(test_data, str):
            if os.path.isdir(test_data) or is_image_file(test_data):
                image_modality_only = True

        return image_modality_only

    def predict(
        self,
        test_data: Union[str, pd.DataFrame],
        test_data_image_column: Optional[str] = None,
        **kwargs,
    ) -> Optional[pd.Series]:
        """
        test_data: str
            The test data to be inferenced.
            Can be a pandas.DataFrame or a local path to a csv file.
            When predicting multimodality with image modality:
                You need to specify `test_data_image_column`, and make sure the image column contains relative path to the image.
            When predicting with only images:
                Can be a local path to a directory containing the images or a local path to a single image.
        test_data_image_column: Optional(str)
            If test_data involves image modality, you must specify the column name corresponding to image paths.
            The path MUST be an abspath
        kwargs:
            Refer to `CloudPredictor.predict()`
        """
        image_modality_only = self._check_image_modality_only(test_data)

        if image_modality_only:
            processed_args = self._prepare_image_predict_args(**kwargs)
            return super().predict(
                test_data,
                test_data_image_column=None,
                split_type=processed_args["split_type"],
                content_type=processed_args["content_type"],
                transformer_kwargs=processed_args["transformer_kwargs"],
                **kwargs,
            )
        else:
            return super().predict(
                test_data,
                test_data_image_column=test_data_image_column,
                **kwargs,
            )

    def predict_proba(
        self,
        test_data: Union[str, pd.DataFrame],
        test_data_image_column: Optional[str] = None,
        **kwargs,
    ) -> Optional[Union[Tuple[pd.Series, Union[pd.DataFrame, pd.Series]], Union[pd.DataFrame, pd.Series]]]:
        """
        test_data: str
            The test data to be inferenced.
            Can be a pandas.DataFrame or a local path to a csv file.
            When predicting multimodality with image modality:
                You need to specify `test_data_image_column`, and make sure the image column contains relative path to the image.
            When predicting with only images:
                Can be a local path to a directory containing the images or a local path to a single image.
        test_data_image_column: Optional(str)
            If test_data involves image modality, you must specify the column name corresponding to image paths.
            The path MUST be an abspath
        kwargs:
            Refer to `CloudPredictor.predict()`
        """
        image_modality_only = self._check_image_modality_only(test_data)

        if image_modality_only:
            processed_args = self._prepare_image_predict_args(**kwargs)
            return super().predict_proba(
                test_data,
                test_data_image_column=None,
                split_type=processed_args["split_type"],
                content_type=processed_args["content_type"],
                transformer_kwargs=processed_args["transformer_kwargs"],
                **kwargs,
            )
        else:
            return super().predict_proba(
                test_data,
                test_data_image_column=test_data_image_column,
                **kwargs,
            )
