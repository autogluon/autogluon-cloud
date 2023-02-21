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
        Predict using SageMaker batch transform.
        When minimizing latency isn't a concern, then the batch transform functionality may be easier, more scalable, and more appropriate.
        If you want to minimize latency, use `predict_real_time()` instead.
        To learn more: https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html
        This method would first create a AutoGluonSagemakerInferenceModel with the trained predictor,
        then create a transformer with it, and call transform in the end.

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
