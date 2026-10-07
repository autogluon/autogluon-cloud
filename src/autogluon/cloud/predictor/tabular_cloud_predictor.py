from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from typing_extensions import Unpack

from ..backend.constant import SAGEMAKER, TABULAR_SAGEMAKER
from ..endpoint.tabular_endpoint import TabularEndpoint
from ..utils.constants import DEFAULT_FRAMEWORK_VERSION
from ..utils.sagemaker_api import (
    IGNORED_TRAINING_KWARGS,
    TrainingJobKwargs,
    check_backend_kwargs,
    reject_legacy_kwargs,
)
from ..utils.utils import split_pred_and_pred_proba
from .cloud_predictor import CloudPredictor

logger = logging.getLogger(__name__)


class TabularCloudPredictor(CloudPredictor[TabularEndpoint]):
    """Train and deploy AutoGluon tabular models (classification and regression) on Amazon SageMaker.

    Wraps :class:`autogluon.tabular.TabularPredictor` (`docs <https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html>`_)
    and runs ``fit``, ``predict``, and endpoint deployment as managed SageMaker jobs.
    """

    predictor_file_name = "TabularCloudPredictor.pkl"
    backend_map = {SAGEMAKER: TABULAR_SAGEMAKER}
    _endpoint_cls = TabularEndpoint

    @property
    def predictor_type(self):
        """
        Type of the underlying AutoGluon predictor.
        """
        return "tabular"

    def _get_local_predictor_cls(self):
        from autogluon.tabular import TabularPredictor

        predictor_cls = TabularPredictor
        return predictor_cls

    @reject_legacy_kwargs
    def fit_predict(
        self,
        train_data: str | Path | pd.DataFrame,
        test_data: str | Path | pd.DataFrame,
        *,
        predictor_init_args: dict[str, Any],
        predictor_fit_args: dict[str, Any] | None = None,
        framework_version: str = DEFAULT_FRAMEWORK_VERSION,
        instance_type: str = "ml.m5.2xlarge",
        wait: bool = True,
        predictions_path: str | None = None,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs: Unpack[TrainingJobKwargs],
    ) -> pd.Series | None:
        """
        Fit and predict in a single SageMaker training job.

        Fits a ``TabularPredictor`` on ``train_data`` and predicts on ``test_data`` in the same job, which is faster
        than :meth:`fit` followed by :meth:`predict`. The predictor stays fitted, so :meth:`deploy` and
        :meth:`predict` still work afterward.

        Parameters
        ----------
        train_data: str | pathlib.Path | pd.DataFrame
            Training data, as a ``pd.DataFrame`` or local/S3 path to a data file.
        test_data: str | pathlib.Path | pd.DataFrame
            Data to predict on, as a ``pd.DataFrame`` or local/S3 path to a data file. Must contain every feature
            column present in ``train_data`` (the label column is not required).
        predictor_init_args: dict
            Arguments forwarded to ``TabularPredictor()``, e.g. ``{"label": "target"}``.
        predictor_fit_args: dict | None, default = None
            Additional fit args forwarded to ``TabularPredictor.fit()``. Must NOT contain ``train_data`` or
            ``tuning_data``.
        framework_version: str, optional
            AutoGluon version, e.g. ``"1.6"``. Training uses the official AutoGluon DLC image for this version.
            Ignored if ``custom_image_uri`` is set.
        instance_type: str, default = "ml.m5.2xlarge"
            Instance type of the training job.
        wait: bool, default = True
            Whether to block until the job completes. If ``False``, returns ``None`` once the job is launched.
        predictions_path: str | None, default = None
            S3 URL of the predictions file, ending in ``.csv`` or ``.parquet``. Defaults to
            ``{cloud_output_path}/{job_name}/predictions.csv``.

        Returns
        -------
        pd.Series | None
            Predictions, or ``None`` if ``wait=False``; fetch them later with :meth:`get_fit_predict_results`.

        Other Parameters
        ----------------
        backend_overrides: dict[str, dict[str, Any]] | None, default = None
            Raw SageMaker request fields for settings without a dedicated argument.

            * Keys: request names from the *SageMaker API* section below.
            * Values: request fields in PascalCase, as in the SageMaker API and boto3. Deep-merged over the request
              built by AutoGluon-Cloud; lists and other non-dict values replace the generated ones.
            * Example: ``{"CreateTrainingJob": {"RetryStrategy": {"MaximumRetryAttempts": 2}}}``
        job_name: str | None, default = None
            Name of the training job. Defaults to a unique name with prefix ``ag-cloud-tabular``.
        volume_size: int, default = 100
            Size in GB of the EBS volume that stores the training data and model artifacts.
        custom_image_uri: str | None, default = None
            Custom training container image URI. If set, ``framework_version`` is ignored.
        timeout: int, default = 86400
            Maximum training job runtime in seconds. Defaults to 24 hours.

        SageMaker API
        -------------
        * :sm-api:`CreateTrainingJob`: trains the predictor and predicts in the same job on ``instance_type``.
          Predictions are written to ``predictions_path``.
        """
        kwargs = check_backend_kwargs(kwargs, TrainingJobKwargs, "fit_predict", IGNORED_TRAINING_KWARGS)
        result = self.fit_predict_proba(
            train_data=train_data,
            test_data=test_data,
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            include_predict=True,
            framework_version=framework_version,
            instance_type=instance_type,
            wait=wait,
            predictions_path=predictions_path,
            backend_overrides=backend_overrides,
            **kwargs,
        )
        if result is None:  # wait=False
            return None
        pred, _ = result
        return pred

    @reject_legacy_kwargs
    def fit_predict_proba(
        self,
        train_data: str | Path | pd.DataFrame,
        test_data: str | Path | pd.DataFrame,
        *,
        predictor_init_args: dict[str, Any],
        predictor_fit_args: dict[str, Any] | None = None,
        include_predict: bool = True,
        framework_version: str = DEFAULT_FRAMEWORK_VERSION,
        instance_type: str = "ml.m5.2xlarge",
        wait: bool = True,
        predictions_path: str | None = None,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs: Unpack[TrainingJobKwargs],
    ) -> tuple[pd.Series, pd.DataFrame | pd.Series] | pd.DataFrame | pd.Series | None:
        """
        Fit and predict probabilities in a single SageMaker training job.

        Same as :meth:`fit_predict`, but returns class probabilities. For regression, the "probabilities" are
        identical to the predictions.

        Parameters
        ----------
        train_data: str | pathlib.Path | pd.DataFrame
            Training data, as a ``pd.DataFrame`` or local/S3 path to a data file.
        test_data: str | pathlib.Path | pd.DataFrame
            Data to predict on, as a ``pd.DataFrame`` or local/S3 path to a data file. Must contain every feature
            column present in ``train_data``.
        predictor_init_args: dict
            Arguments forwarded to ``TabularPredictor()``, e.g. ``{"label": "target"}``.
        predictor_fit_args: dict | None, default = None
            Additional fit args forwarded to ``TabularPredictor.fit()``. Must NOT contain ``train_data`` or
            ``tuning_data``.
        include_predict: bool, default = True
            Whether to also return the predictions. The job always computes both, so this adds no cost.
        framework_version: str, optional
            AutoGluon version, e.g. ``"1.6"``. Training uses the official AutoGluon DLC image for this version.
            Ignored if ``custom_image_uri`` is set.
        instance_type: str, default = "ml.m5.2xlarge"
            Instance type of the training job.
        wait: bool, default = True
            Whether to block until the job completes. If ``False``, returns ``None`` once the job is launched.
        predictions_path: str | None, default = None
            S3 URL of the predictions file, ending in ``.csv`` or ``.parquet``. Defaults to
            ``{cloud_output_path}/{job_name}/predictions.csv``.

        Returns
        -------
        tuple[pd.Series, pd.DataFrame | pd.Series] | pd.DataFrame | pd.Series | None
            ``(prediction, predict_probability)`` if ``include_predict=True``, otherwise ``predict_probability``.
            Returns ``None`` if ``wait=False``; fetch them later with :meth:`get_fit_predict_proba_results`.

        Other Parameters
        ----------------
        backend_overrides: dict[str, dict[str, Any]] | None, default = None
            Raw SageMaker request fields for settings without a dedicated argument.

            * Keys: request names from the *SageMaker API* section below.
            * Values: request fields in PascalCase, as in the SageMaker API and boto3. Deep-merged over the request
              built by AutoGluon-Cloud; lists and other non-dict values replace the generated ones.
            * Example: ``{"CreateTrainingJob": {"RetryStrategy": {"MaximumRetryAttempts": 2}}}``
        job_name: str | None, default = None
            Name of the training job. Defaults to a unique name with prefix ``ag-cloud-tabular``.
        volume_size: int, default = 100
            Size in GB of the EBS volume that stores the training data and model artifacts.
        custom_image_uri: str | None, default = None
            Custom training container image URI. If set, ``framework_version`` is ignored.
        timeout: int, default = 86400
            Maximum training job runtime in seconds. Defaults to 24 hours.

        SageMaker API
        -------------
        * :sm-api:`CreateTrainingJob`: trains the predictor and predicts in the same job on ``instance_type``.
          Predictions are written to ``predictions_path``.
        """
        kwargs = check_backend_kwargs(kwargs, TrainingJobKwargs, "fit_predict_proba", IGNORED_TRAINING_KWARGS)
        extra_ag_args = {"predict_after_fit": True}
        if predictions_path is not None:
            extra_ag_args["predictions_path"] = predictions_path

        self._fit(
            data_channels={"train_data": train_data, "tuning_data": None, "test_data": test_data},
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            framework_version=framework_version,
            instance_type=instance_type,
            wait=wait,
            backend_overrides=backend_overrides,
            extra_ag_args=extra_ag_args,
            **kwargs,
        )

        if not wait:
            logger.info(
                "fit_predict job launched asynchronously. Use `get_fit_job_status()` to poll, then "
                "`get_fit_predict_results()` / `get_fit_predict_proba_results()` to fetch the results."
            )
            return None

        pred, pred_proba = self.get_fit_predict_proba_results()
        if include_predict:
            return pred, pred_proba
        return pred_proba

    def get_fit_predict_results(self) -> pd.Series:
        """
        Retrieve the predictions of a completed :meth:`fit_predict` or :meth:`fit_predict_proba` job.

        Returns
        -------
        pd.Series
            Predictions for ``test_data``.
        """
        pred, _ = self.get_fit_predict_proba_results()
        return pred

    def get_fit_predict_proba_results(self) -> tuple[pd.Series, pd.DataFrame | pd.Series]:
        """
        Retrieve the predictions and probabilities of a completed :meth:`fit_predict` or :meth:`fit_predict_proba` job.

        Returns
        -------
        tuple[pd.Series, pd.DataFrame | pd.Series]
            ``(prediction, predict_probability)``. For regression the probabilities are identical to the
            predictions.
        """
        raw = self.backend.get_fit_predict_results()
        pred, pred_proba = split_pred_and_pred_proba(raw)
        # Regression: the job writes only the prediction column, so proba mirrors pred (matches predict_proba).
        if pred_proba is None:
            pred_proba = pred
        return pred, pred_proba
