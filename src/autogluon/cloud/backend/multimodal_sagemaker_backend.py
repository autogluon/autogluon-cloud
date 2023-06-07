import copy
import os
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
from sagemaker import Predictor

from autogluon.common.loaders import load_pd

from ..utils.ag_sagemaker import AutoGluonMultiModalRealtimePredictor
from ..utils.utils import convert_image_path_to_encoded_bytes_in_dataframe, is_image_file, read_image_bytes_and_encode
from .constant import MULTIMODL_SAGEMAKER
from .sagemaker_backend import SagemakerBackend


class MultiModalSagemakerBackend(SagemakerBackend):
    name = MULTIMODL_SAGEMAKER

    @property
    def _realtime_predictor_cls(self) -> Predictor:
        """Class used for realtime endpoint"""
        return AutoGluonMultiModalRealtimePredictor

    def _load_predict_real_time_test_data(
        self, test_data: Union[str, pd.DataFrame], test_data_image_column: str
    ) -> Tuple[pd.DataFrame, str]:
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
        test_data: Union[str, pd.DataFrame],
        test_data_image_column: Optional[str] = None,
        accept: str = "application/x-parquet",
        inference_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
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
        inference_kwargs: Optional[Dict[str, Any]], default = None
            Additional args that you would pass to `predict` calls of an AutoGluon logic

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
        pred, _ = self._predict_real_time(
            test_data=test_data, accept=accept, inference_kwargs=inference_kwargs, ContentType=content_type
        )

        return pred

    def predict_proba_real_time(
        self,
        test_data: Union[str, pd.DataFrame],
        test_data_image_column: Optional[str] = None,
        accept: str = "application/x-parquet",
        inference_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
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
        inference_kwargs: Optional[Dict[str, Any]], default = None
            Additional args that you would pass to `predict` calls of an AutoGluon logic

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
        pred, proba = self._predict_real_time(
            test_data=test_data, accept=accept, inference_kwargs=inference_kwargs, ContentType=content_type
        )

        if proba is None:
            return pred

        return proba

    def predict(
        self,
        test_data: Union[str, pd.DataFrame],
        test_data_image_column: Optional[str] = None,
        **kwargs,
    ) -> Optional[pd.Series]:
        """
        Predict using SageMaker batch transform.
        When minimizing latency isn't a concern, then the batch transform functionality may be easier, more scalable, and more appropriate.
        If you want to minimize latency, use `predict_real_time()` instead.
        To learn more: https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html
        This method would first create a AutoGluonSagemakerInferenceModel with the trained predictor,
        then create a transformer with it, and call transform in the end.

        Parameters
        ----------
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
            Refer to `SagemakerBackend.predict()`
        """
        image_modality_only = self._check_image_modality_only(test_data)

        if image_modality_only:
            processed_args = self._prepare_image_predict_args(**kwargs)
            kwargs["transformer_kwargs"] = processed_args["transformer_kwargs"]
            kwargs["transform_kwargs"] = processed_args["transform_kwargs"]
            return super().predict(test_data, test_data_image_column=None, **kwargs)
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
    ) -> Optional[pd.Series]:
        """
        Predict proba using SageMaker batch transform.
        When minimizing latency isn't a concern, then the batch transform functionality may be easier, more scalable, and more appropriate.
        If you want to minimize latency, use `predict_real_time()` instead.
        To learn more: https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html
        This method would first create a AutoGluonSagemakerInferenceModel with the trained predictor,
        then create a transformer with it, and call transform in the end.

        Parameters
        ----------
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
            Refer to `SagemakerBackend.predict_proba()`
        """
        image_modality_only = self._check_image_modality_only(test_data)

        if image_modality_only:
            processed_args = self._prepare_image_predict_args(**kwargs)
            kwargs["transformer_kwargs"] = processed_args["transformer_kwargs"]
            kwargs["transform_kwargs"] = processed_args["transform_kwargs"]
            return super().predict_proba(test_data, test_data_image_column=None, **kwargs)
        else:
            return super().predict_proba(
                test_data,
                test_data_image_column=test_data_image_column,
                **kwargs,
            )

    def _prepare_image_predict_args(self, **predict_kwargs):
        split_type = None
        content_type = "application/x-image"
        predict_kwargs = copy.deepcopy(predict_kwargs)
        transformer_kwargs = predict_kwargs.pop("transformer_kwargs", {})
        if transformer_kwargs is None:
            transformer_kwargs = {}
        transformer_kwargs["strategy"] = "SingleRecord"
        transform_kwargs = predict_kwargs.pop("transofrm_kwargs", {})
        if transform_kwargs is None:
            transform_kwargs = {}
        transform_kwargs["split_type"] = split_type
        transform_kwargs["content_type"] = content_type

        return {"transformer_kwargs": transformer_kwargs, "transform_kwargs": transform_kwargs}

    def _check_image_modality_only(self, test_data):
        image_modality_only = False
        if isinstance(test_data, str):
            if os.path.isdir(test_data) or is_image_file(test_data):
                image_modality_only = True

        return image_modality_only
