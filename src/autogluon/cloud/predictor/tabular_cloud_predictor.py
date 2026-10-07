from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ..backend.constant import SAGEMAKER, TABULAR_SAGEMAKER
from ..endpoint.tabular_endpoint import TabularEndpoint
from ..utils.constants import DEFAULT_FRAMEWORK_VERSION, DEFAULT_VOLUME_SIZE
from ..utils.sagemaker_api import reject_legacy_kwargs
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
        leaderboard: bool = True,
        framework_version: str = DEFAULT_FRAMEWORK_VERSION,
        job_name: str | None = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        volume_size: int = DEFAULT_VOLUME_SIZE,
        custom_image_uri: str | None = None,
        wait: bool = True,
        predictions_path: str | None = None,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
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
        leaderboard: bool, default = True
            Whether to include the leaderboard in the output artifact.
        framework_version: str, optional
            AutoGluon version, e.g. ``"1.6"``. Training uses the official AutoGluon DLC image for this version.
            Ignored if ``custom_image_uri`` is set.
        job_name: str | None, default = None
            Name of the training job. If ``None``, a unique name with prefix ``ag-cloud-tabular`` is generated.
        instance_type: str, default = "ml.m5.2xlarge"
            Instance type of the training job.
        instance_count: int, default = 1
            Number of training instances. Only single-instance training is supported.
        volume_size: int, default = 100
            Size in GB of the EBS volume that stores the training data and model artifacts.
        custom_image_uri: str | None, default = None
            Custom training container image URI. If set, ``framework_version`` is ignored.
        wait: bool, default = True
            Whether to block until the job completes. If ``False``, returns ``None`` once the job is launched.
        predictions_path: str | None, default = None
            S3 URL of the predictions file, ending in ``.csv`` or ``.parquet``. Defaults to
            ``{cloud_output_path}/{job_name}/predictions.csv``.
        backend_overrides: dict[str, dict[str, Any]] | None, default = None
            Raw SageMaker request fields for settings without a dedicated argument.

            * Keys: request names from the *SageMaker API* section below.
            * Values: request fields in PascalCase, as in the SageMaker API and boto3. Deep-merged over the request
              built by AutoGluon-Cloud; lists and other non-dict values replace the generated ones.
            * Example: ``{"CreateTrainingJob": {"RetryStrategy": {"MaximumRetryAttempts": 2}}}``

        Returns
        -------
        pd.Series | None
            Predictions, or ``None`` if ``wait=False``; fetch them later with :meth:`get_fit_predict_results`.

        SageMaker API
        -------------
        * :sm-api:`CreateTrainingJob`: trains the predictor and predicts in the same job on ``instance_count`` x
          ``instance_type``. Predictions are written to ``predictions_path``.
        """
        result = self.fit_predict_proba(
            train_data=train_data,
            test_data=test_data,
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            include_predict=True,
            leaderboard=leaderboard,
            framework_version=framework_version,
            job_name=job_name,
            instance_type=instance_type,
            instance_count=instance_count,
            volume_size=volume_size,
            custom_image_uri=custom_image_uri,
            wait=wait,
            predictions_path=predictions_path,
            backend_overrides=backend_overrides,
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
        leaderboard: bool = True,
        framework_version: str = DEFAULT_FRAMEWORK_VERSION,
        job_name: str | None = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        volume_size: int = DEFAULT_VOLUME_SIZE,
        custom_image_uri: str | None = None,
        wait: bool = True,
        predictions_path: str | None = None,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
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
        leaderboard: bool, default = True
            Whether to include the leaderboard in the output artifact.
        framework_version: str, optional
            AutoGluon version, e.g. ``"1.6"``. Training uses the official AutoGluon DLC image for this version.
            Ignored if ``custom_image_uri`` is set.
        job_name: str | None, default = None
            Name of the training job. If ``None``, a unique name with prefix ``ag-cloud-tabular`` is generated.
        instance_type: str, default = "ml.m5.2xlarge"
            Instance type of the training job.
        instance_count: int, default = 1
            Number of training instances. Only single-instance training is supported.
        volume_size: int, default = 100
            Size in GB of the EBS volume that stores the training data and model artifacts.
        custom_image_uri: str | None, default = None
            Custom training container image URI. If set, ``framework_version`` is ignored.
        wait: bool, default = True
            Whether to block until the job completes. If ``False``, returns ``None`` once the job is launched.
        predictions_path: str | None, default = None
            S3 URL of the predictions file, ending in ``.csv`` or ``.parquet``. Defaults to
            ``{cloud_output_path}/{job_name}/predictions.csv``.
        backend_overrides: dict[str, dict[str, Any]] | None, default = None
            Raw SageMaker request fields for settings without a dedicated argument.

            * Keys: request names from the *SageMaker API* section below.
            * Values: request fields in PascalCase, as in the SageMaker API and boto3. Deep-merged over the request
              built by AutoGluon-Cloud; lists and other non-dict values replace the generated ones.
            * Example: ``{"CreateTrainingJob": {"RetryStrategy": {"MaximumRetryAttempts": 2}}}``

        Returns
        -------
        tuple[pd.Series, pd.DataFrame | pd.Series] | pd.DataFrame | pd.Series | None
            ``(prediction, predict_probability)`` if ``include_predict=True``, otherwise ``predict_probability``.
            Returns ``None`` if ``wait=False``; fetch them later with :meth:`get_fit_predict_proba_results`.

        SageMaker API
        -------------
        * :sm-api:`CreateTrainingJob`: trains the predictor and predicts in the same job on ``instance_count`` x
          ``instance_type``. Predictions are written to ``predictions_path``.
        """
        extra_ag_args = {"predict_after_fit": True}
        if predictions_path is not None:
            extra_ag_args["predictions_path"] = predictions_path

        self.fit(
            train_data=train_data,
            test_data=test_data,
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            leaderboard=leaderboard,
            framework_version=framework_version,
            job_name=job_name,
            instance_type=instance_type,
            instance_count=instance_count,
            volume_size=volume_size,
            custom_image_uri=custom_image_uri,
            wait=wait,
            backend_overrides=backend_overrides,
            extra_ag_args=extra_ag_args,
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
