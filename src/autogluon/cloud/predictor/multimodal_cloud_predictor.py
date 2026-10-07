from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from typing_extensions import Self, Unpack

from ..backend.constant import MULTIMODL_SAGEMAKER, SAGEMAKER
from ..endpoint.multimodal_endpoint import MultiModalEndpoint
from ..utils.constants import DEFAULT_FRAMEWORK_VERSION
from ..utils.sagemaker_api import (
    IGNORED_TRAINING_KWARGS,
    BatchTransformKwargs,
    TrainingJobKwargs,
    check_backend_kwargs,
    reject_legacy_kwargs,
)
from .cloud_predictor import CloudPredictor

logger = logging.getLogger(__name__)


class MultiModalCloudPredictor(CloudPredictor[MultiModalEndpoint]):
    """Train and deploy AutoGluon multimodal models (image, text, tabular) on Amazon SageMaker.

    Wraps :class:`autogluon.multimodal.MultiModalPredictor` (`docs <https://auto.gluon.ai/stable/api/autogluon.multimodal.MultiModalPredictor.html>`_)
    and runs ``fit``, ``predict``, and endpoint deployment as managed SageMaker jobs.
    """

    predictor_file_name = "MultiModalCloudPredictor.pkl"
    backend_map = {SAGEMAKER: MULTIMODL_SAGEMAKER}
    _endpoint_cls = MultiModalEndpoint

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "AutoGluon Multimodal is on a deprecation path. "
            "MultiModalCloudPredictor will be removed in a future release.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    @property
    def predictor_type(self) -> str:
        """
        Type of the underneath AutoGluon Predictor
        """
        return "multimodal"

    def _get_local_predictor_cls(self):
        from autogluon.multimodal import MultiModalPredictor

        predictor_cls = MultiModalPredictor
        return predictor_cls

    @reject_legacy_kwargs
    def fit(
        self,
        train_data: str | Path | pd.DataFrame | None = None,
        *,
        tuning_data: str | Path | pd.DataFrame | None = None,
        predictor_init_args: dict[str, Any],
        predictor_fit_args: dict[str, Any] | None = None,
        image_column: str | None = None,
        framework_version: str = DEFAULT_FRAMEWORK_VERSION,
        instance_type: str = "ml.m5.2xlarge",
        wait: bool = True,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs: Unpack[TrainingJobKwargs],
    ) -> Self:
        """
        Fit the predictor in a SageMaker training job.

        Same as :meth:`CloudPredictor.fit`, with one additional argument:

        Parameters
        ----------
        image_column: str | None, default = None
            Name of the column containing absolute local paths to images, if the data contains images.
        """
        kwargs = check_backend_kwargs(kwargs, TrainingJobKwargs, "fit", IGNORED_TRAINING_KWARGS)
        self._fit(
            data_channels={"train_data": train_data, "tuning_data": tuning_data},
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            image_column=image_column,
            framework_version=framework_version,
            instance_type=instance_type,
            wait=wait,
            backend_overrides=backend_overrides,
            **kwargs,
        )
        return self

    @reject_legacy_kwargs
    def predict(
        self,
        test_data: str | pd.DataFrame,
        *,
        test_data_image_column: str | None = None,
        predictor_path: str | None = None,
        framework_version: str | None = None,
        instance_type: str = "ml.m5.2xlarge",
        wait: bool = True,
        predictions_path: str | None = None,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs: Unpack[BatchTransformKwargs],
    ) -> pd.Series | None:
        """
        Predict with a SageMaker batch transform job.

        Same as :meth:`CloudPredictor.predict`, with one additional argument:

        Parameters
        ----------
        test_data_image_column: str | None, default = None
            Name of the column containing absolute paths to images, if the data contains images.
        """
        kwargs = check_backend_kwargs(kwargs, BatchTransformKwargs, "predict")
        return self.backend.predict(
            test_data=test_data,
            test_data_image_column=test_data_image_column,
            predictor_path=predictor_path,
            framework_version=framework_version,
            instance_type=instance_type,
            wait=wait,
            predictions_path=predictions_path,
            backend_overrides=backend_overrides,
            **kwargs,
        )

    @reject_legacy_kwargs
    def predict_proba(
        self,
        test_data: str | pd.DataFrame,
        *,
        test_data_image_column: str | None = None,
        include_predict: bool = True,
        predictor_path: str | None = None,
        framework_version: str | None = None,
        instance_type: str = "ml.m5.2xlarge",
        wait: bool = True,
        predictions_path: str | None = None,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs: Unpack[BatchTransformKwargs],
    ) -> tuple[pd.Series, pd.DataFrame | pd.Series] | pd.DataFrame | pd.Series | None:
        """
        Predict class probabilities with a SageMaker batch transform job.

        Same as :meth:`CloudPredictor.predict_proba`, with one additional argument:

        Parameters
        ----------
        test_data_image_column: str | None, default = None
            Name of the column containing absolute paths to images, if the data contains images.
        """
        kwargs = check_backend_kwargs(kwargs, BatchTransformKwargs, "predict_proba")
        return self.backend.predict_proba(
            test_data=test_data,
            test_data_image_column=test_data_image_column,
            include_predict=include_predict,
            predictor_path=predictor_path,
            framework_version=framework_version,
            instance_type=instance_type,
            wait=wait,
            predictions_path=predictions_path,
            backend_overrides=backend_overrides,
            **kwargs,
        )
