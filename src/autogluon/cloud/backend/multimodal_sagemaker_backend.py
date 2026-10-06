import os
from typing import Any

import pandas as pd

from autogluon.common.loaders import load_pd

from ..utils.serializers import MultiModalSerializer
from ..utils.utils import convert_image_path_to_encoded_bytes_in_dataframe, is_image_file, read_image_bytes_and_encode
from .constant import MULTIMODL_SAGEMAKER
from .sagemaker_backend import SagemakerBackend


class MultiModalSagemakerBackend(SagemakerBackend):
    name = MULTIMODL_SAGEMAKER
    # Images are sent one file per request.
    _IMAGE_BATCH_ARGS = dict(content_type="application/x-image", split_type=None, batch_strategy="SingleRecord")

    def _realtime_serializer(self):
        """Serializer used for realtime endpoint requests"""
        return MultiModalSerializer()

    def _load_predict_real_time_test_data(
        self, test_data: str | pd.DataFrame, test_data_image_column: str
    ) -> tuple[pd.DataFrame, str]:
        import numpy as np

        if isinstance(test_data, str):
            if is_image_file(test_data):
                test_data = [test_data]
            else:
                test_data = load_pd.load(test_data)
        if isinstance(test_data, list):
            test_data = np.array([read_image_bytes_and_encode(image) for image in test_data], dtype="object")
            content_type = "application/x-autogluon-npy"
        if isinstance(test_data, pd.DataFrame):
            if test_data_image_column is not None:
                test_data = convert_image_path_to_encoded_bytes_in_dataframe(
                    dataframe=test_data, image_column=test_data_image_column
                )
            content_type = "application/x-autogluon-parquet"

        return test_data, content_type

    def predict_real_time(
        self,
        test_data: str | pd.DataFrame,
        test_data_image_column: str | None = None,
        accept: str = "application/x-parquet",
        inference_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> pd.Series:
        """
        Predict with the deployed SageMaker endpoint. A deployed SageMaker endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict()` instead.

        Parameters
        ----------
        test_data: str | pd.DataFrame
            The test data to be inferenced.
            Can be a ``pd.DataFrame`` or a local path to a csv file.
            When predicting multimodality with image modality:
                You need to specify `test_data_image_column`, and make sure the image column contains relative path to the image.
            When predicting with only images:
                Can be a ``pd.DataFrame`` or a local path to a csv file.
                    Similarly, you need to specify `test_data_image_column`, and make sure the image column contains relative path to the image.
                Or a local path to a single image file.
                Or a list of local paths to image files.
        test_data_image_column: str | None, default = None
            If provided a csv file or ``pd.DataFrame`` as the test_data and test_data involves image modality,
            you must specify the column name corresponding to image paths.
            The path MUST be an abspath
        accept: str, default = application/x-parquet
            Type of accept output content.
            Valid options are application/x-parquet, text/csv, application/json
        inference_kwargs: dict[str, Any] | None, default = None
            Additional args that you would pass to `predict` calls of an AutoGluon logic

        Returns
        -------
        pd.Series
            Predict results in ``pd.Series``
        """
        self._validate_predict_real_time_args(accept)
        test_data, content_type = self._load_predict_real_time_test_data(
            test_data=test_data, test_data_image_column=test_data_image_column
        )
        # The serializer's content type is fixed, so the per-request content type is passed explicitly.
        pred, _ = self._predict_real_time(
            test_data=test_data, accept=accept, inference_kwargs=inference_kwargs, content_type=content_type
        )

        return pred

    def predict_proba_real_time(
        self,
        test_data: str | pd.DataFrame,
        test_data_image_column: str | None = None,
        accept: str = "application/x-parquet",
        inference_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> pd.DataFrame | pd.Series:
        """
        Predict with the deployed SageMaker endpoint. A deployed SageMaker endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict()` instead.

        Parameters
        ----------
        test_data: str | pd.DataFrame
            The test data to be inferenced.
            Can be a ``pd.DataFrame`` or a local path to a csv file.
            When predicting multimodality with image modality:
                You need to specify `test_data_image_column`, and make sure the image column contains relative path to the image.
            When predicting with only images:
                Can be a ``pd.DataFrame`` or a local path to a csv file.
                    Similarly, you need to specify `test_data_image_column`, and make sure the image column contains relative path to the image.
                Or a local path to a single image file.
                Or a list of local paths to image files.
        test_data_image_column: str | None, default = None
            If provided a csv file or ``pd.DataFrame`` as the test_data and test_data involves image modality,
            you must specify the column name corresponding to image paths.
            The path MUST be an abspath
        accept: str, default = application/x-parquet
            Type of accept output content.
            Valid options are application/x-parquet, text/csv, application/json
        inference_kwargs: dict[str, Any] | None, default = None
            Additional args that you would pass to `predict` calls of an AutoGluon logic

        Returns
        -------
        pd.DataFrame | pd.Series
            Will return a ``pd.Series`` when it's a regression problem. Will return a ``pd.DataFrame`` otherwise
        """
        self._validate_predict_real_time_args(accept)
        test_data, content_type = self._load_predict_real_time_test_data(
            test_data=test_data, test_data_image_column=test_data_image_column
        )
        # The serializer's content type is fixed, so the per-request content type is passed explicitly.
        pred, proba = self._predict_real_time(
            test_data=test_data, accept=accept, inference_kwargs=inference_kwargs, content_type=content_type
        )

        if proba is None:
            return pred

        return proba

    def predict(
        self,
        test_data: str | pd.DataFrame,
        test_data_image_column: str | None = None,
        **kwargs,
    ) -> pd.Series | None:
        """
        Predict using SageMaker batch transform.
        When minimizing latency isn't a concern, then the batch transform functionality may be easier, more scalable, and more appropriate.
        If you want to minimize latency, deploy an endpoint with `deploy()` instead.
        To learn more: https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html

        Parameters
        ----------
        test_data: str | pd.DataFrame
            The test data to be inferenced.
            Can be a ``pd.DataFrame`` or a local path to a csv file.
            When predicting multimodality with image modality:
                You need to specify `test_data_image_column`, and make sure the image column contains relative path to the image.
            When predicting with only images:
                Can be a local path to a directory containing the images or a local path to a single image.
        test_data_image_column: str | None
            If test_data involves image modality, you must specify the column name corresponding to image paths.
            The path MUST be an abspath
        **kwargs: Any
            Refer to `SagemakerBackend.predict()`
        """
        image_modality_only = self._check_image_modality_only(test_data)

        if image_modality_only:
            pred, _ = self._predict(
                test_data, original_features=self.original_features, **kwargs, **self._IMAGE_BATCH_ARGS
            )
            return pred
        else:
            return super().predict(
                test_data,
                test_data_image_column=test_data_image_column,
                **kwargs,
            )

    def predict_proba(
        self,
        test_data: str | pd.DataFrame,
        test_data_image_column: str | None = None,
        **kwargs,
    ) -> pd.Series | None:
        """
        Predict proba using SageMaker batch transform.
        When minimizing latency isn't a concern, then the batch transform functionality may be easier, more scalable, and more appropriate.
        If you want to minimize latency, deploy an endpoint with `deploy()` instead.
        To learn more: https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html

        Parameters
        ----------
        test_data: str | pd.DataFrame
            The test data to be inferenced.
            Can be a ``pd.DataFrame`` or a local path to a csv file.
            When predicting multimodality with image modality:
                You need to specify `test_data_image_column`, and make sure the image column contains relative path to the image.
            When predicting with only images:
                Can be a local path to a directory containing the images or a local path to a single image.
        test_data_image_column: str | None
            If test_data involves image modality, you must specify the column name corresponding to image paths.
            The path MUST be an abspath
        **kwargs: Any
            Refer to `SagemakerBackend.predict_proba()`
        """
        image_modality_only = self._check_image_modality_only(test_data)

        if image_modality_only:
            include_predict = kwargs.pop("include_predict", True)
            pred, pred_proba = self._predict(
                test_data, original_features=self.original_features, **kwargs, **self._IMAGE_BATCH_ARGS
            )
            return (pred, pred_proba) if include_predict else pred_proba
        else:
            return super().predict_proba(
                test_data,
                test_data_image_column=test_data_image_column,
                **kwargs,
            )

    def _check_image_modality_only(self, test_data):
        image_modality_only = False
        if isinstance(test_data, str):
            if os.path.isdir(test_data) or is_image_file(test_data):
                image_modality_only = True

        return image_modality_only
