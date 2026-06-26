from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from ..backend.constant import RAY_AWS, SAGEMAKER, TABULAR_RAY_AWS, TABULAR_SAGEMAKER
from ..utils.utils import split_pred_and_pred_proba
from .cloud_predictor import CloudPredictor

logger = logging.getLogger(__name__)


class TabularCloudPredictor(CloudPredictor):
    """Train and deploy AutoGluon tabular models (classification and regression) on AWS SageMaker.

    Wraps :class:`autogluon.tabular.TabularPredictor` (`docs <https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html>`_)
    and runs ``fit``, ``predict``, and endpoint deployment as managed SageMaker jobs.
    """

    predictor_file_name = "TabularCloudPredictor.pkl"
    backend_map = {SAGEMAKER: TABULAR_SAGEMAKER, RAY_AWS: TABULAR_RAY_AWS}

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

    def fit_predict(
        self,
        train_data: Union[str, Path, pd.DataFrame],
        test_data: Union[str, Path, pd.DataFrame],
        *,
        predictor_init_args: Dict[str, Any],
        predictor_fit_args: Optional[Dict[str, Any]] = None,
        leaderboard: bool = True,
        framework_version: str = "latest",
        job_name: Optional[str] = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        volume_size: int = 256,
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        predictions_path: Optional[str] = None,
        backend_kwargs: Optional[Dict] = None,
    ) -> Optional[pd.Series]:
        """
        Fit and predict in a single SageMaker training job.

        Fits a ``TabularPredictor`` on ``train_data`` and runs batch prediction on ``test_data`` inside the same
        training container. This avoids the overhead of a separate batch-transform job (one cold start, one data
        upload, no predictor-tarball round-trip). The predictor is left fitted afterward, so ``deploy()`` /
        ``predict()`` still work.

        Parameters
        ----------
        train_data: Union[str, pathlib.Path, pd.DataFrame]
            Training data, as a DataFrame or local/S3 path to a data file.
        test_data: Union[str, pathlib.Path, pd.DataFrame]
            Data to predict on, as a DataFrame or local/S3 path to a data file. Must contain every feature
            column present in ``train_data`` (the label column is not required).
        predictor_init_args: dict
            Init args for the predictor.
        predictor_fit_args: Optional[dict], default = None
            Additional fit args forwarded to ``TabularPredictor.fit()``. Must NOT contain ``train_data`` or
            ``tuning_data``.
        leaderboard: bool, default = True
            Whether to include the leaderboard in the output artifact.
        framework_version: str, default = `latest`
            Training container version of autogluon. If `latest`, will use the latest available container version.
            If `custom_image_uri` is set, this argument will be ignored.
        job_name: str, default = None
            Name of the launched training job. If None, CloudPredictor will create one with prefix ag-cloudpredictor.
        instance_type: str, default = 'ml.m5.2xlarge'
            Instance type the predictor will be trained on with SageMaker.
        instance_count: int, default = 1
            Number of instances used to fit the predictor.
        volume_size: int, default = 256
            Size in GB of the EBS volume to use for storing input data during training.
        custom_image_uri: Optional[str], default = None
            Custom container image URI. If set, ``framework_version`` is ignored.
        wait: bool, default = True
            Whether the call should wait until the job completes.
        predictions_path: Optional[str]
            S3 URL where predictions will be written by the training container (e.g.
            ``s3://my-bucket/runs/2024-05-01/predictions.csv``). Defaults to
            ``{cloud_output_path}/{job_name}/predictions.csv``.
        backend_kwargs: Optional[dict], default = None
            Backend-specific arguments. Same keys as ``fit()``.

        Returns
        -------
        Optional[pd.Series]
            Predictions as a Series. Returns ``None`` when ``wait`` is False; fetch later via
            ``get_fit_predict_results()``.
        """
        self._fit_for_predict(
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
            predictions_path=predictions_path,
            backend_kwargs=backend_kwargs,
        )

        if not wait:
            logger.info(
                "fit_predict job launched asynchronously. Use `get_fit_job_status()` "
                "to poll, then `get_fit_predict_results()` to fetch predictions."
            )
            return None

        return self.get_fit_predict_results()

    def fit_predict_proba(
        self,
        train_data: Union[str, Path, pd.DataFrame],
        test_data: Union[str, Path, pd.DataFrame],
        *,
        predictor_init_args: Dict[str, Any],
        predictor_fit_args: Optional[Dict[str, Any]] = None,
        include_predict: bool = True,
        leaderboard: bool = True,
        framework_version: str = "latest",
        job_name: Optional[str] = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        volume_size: int = 256,
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        predictions_path: Optional[str] = None,
        backend_kwargs: Optional[Dict] = None,
    ) -> Optional[Union[Tuple[pd.Series, Union[pd.DataFrame, pd.Series]], Union[pd.DataFrame, pd.Series]]]:
        """
        Fit and predict probabilities in a single SageMaker training job.

        Identical to :meth:`fit_predict` but returns class probabilities. For regression the probabilities are
        identical to the predictions (same as :meth:`predict_proba`).

        Parameters
        ----------
        train_data: Union[str, pathlib.Path, pd.DataFrame]
            Training data, as a DataFrame or local/S3 path to a data file.
        test_data: Union[str, pathlib.Path, pd.DataFrame]
            Data to predict on. Must contain every feature column present in ``train_data``.
        predictor_init_args: dict
            Init args for the predictor.
        predictor_fit_args: Optional[dict], default = None
            Additional fit args forwarded to ``TabularPredictor.fit()``.
        include_predict: bool, default = True
            Whether to return the predictions along with the probabilities. Comes for free — the job always
            computes both.
        leaderboard: bool, default = True
            Whether to include the leaderboard in the output artifact.
        framework_version: str, default = `latest`
            Training container version of autogluon. If `custom_image_uri` is set, this argument is ignored.
        job_name: str, default = None
            Name of the launched training job. If None, CloudPredictor will create one with prefix ag-cloudpredictor.
        instance_type: str, default = 'ml.m5.2xlarge'
            Instance type the predictor will be trained on with SageMaker.
        instance_count: int, default = 1
            Number of instances used to fit the predictor.
        volume_size: int, default = 256
            Size in GB of the EBS volume to use for storing input data during training.
        custom_image_uri: Optional[str], default = None
            Custom container image URI. If set, ``framework_version`` is ignored.
        wait: bool, default = True
            Whether the call should wait until the job completes.
        predictions_path: Optional[str]
            S3 URL where predictions will be written by the training container. Defaults to
            ``{cloud_output_path}/{job_name}/predictions.csv``.
        backend_kwargs: Optional[dict], default = None
            Backend-specific arguments. Same keys as ``fit()``.

        Returns
        -------
        Optional[Union[Tuple[pd.Series, Union[pd.DataFrame, pd.Series]], Union[pd.DataFrame, pd.Series]]]
            If ``include_predict`` is True, returns ``(prediction, predict_probability)``; otherwise just
            ``predict_probability``. Returns ``None`` when ``wait`` is False; fetch later via
            ``get_fit_predict_proba_results()``.
        """
        self._fit_for_predict(
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
            predictions_path=predictions_path,
            backend_kwargs=backend_kwargs,
        )

        if not wait:
            logger.info(
                "fit_predict_proba job launched asynchronously. Use `get_fit_job_status()` "
                "to poll, then `get_fit_predict_proba_results()` to fetch probabilities."
            )
            return None

        return self.get_fit_predict_proba_results(include_predict=include_predict)

    def _fit_for_predict(
        self,
        *,
        train_data,
        test_data,
        predictor_init_args,
        predictor_fit_args,
        leaderboard,
        framework_version,
        job_name,
        instance_type,
        instance_count,
        volume_size,
        custom_image_uri,
        wait,
        predictions_path,
        backend_kwargs,
    ) -> None:
        """Launch a fit + in-job-predict training job shared by ``fit_predict`` / ``fit_predict_proba``."""
        backend_kwargs = {} if backend_kwargs is None else dict(backend_kwargs)
        extra_ag_args = dict(backend_kwargs.get("extra_ag_args") or {})
        extra_ag_args["predict_after_fit"] = True
        if predictions_path is not None:
            extra_ag_args["predictions_path"] = predictions_path
        backend_kwargs["extra_ag_args"] = extra_ag_args
        backend_kwargs["test_data"] = test_data

        self.fit(
            train_data=train_data,
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
            backend_kwargs=backend_kwargs,
        )

    def get_fit_predict_results(self) -> pd.Series:
        """
        Retrieve predictions produced by a completed ``fit_predict`` job.

        Returns
        -------
        pd.Series
            Predictions for ``test_data``.
        """
        raw = self.backend.get_fit_predict_results()
        pred, _ = split_pred_and_pred_proba(raw)
        return pred

    def get_fit_predict_proba_results(
        self, include_predict: bool = True
    ) -> Union[Tuple[pd.Series, Union[pd.DataFrame, pd.Series]], Union[pd.DataFrame, pd.Series]]:
        """
        Retrieve probabilities produced by a completed ``fit_predict_proba`` job.

        Parameters
        ----------
        include_predict: bool, default = True
            Whether to return the predictions along with the probabilities.

        Returns
        -------
        Union[Tuple[pd.Series, Union[pd.DataFrame, pd.Series]], Union[pd.DataFrame, pd.Series]]
            If ``include_predict`` is True, returns ``(prediction, predict_probability)``; otherwise just
            ``predict_probability``. For regression the probabilities are identical to the predictions.
        """
        raw = self.backend.get_fit_predict_results()
        pred, pred_proba = split_pred_and_pred_proba(raw)
        # Regression: the job writes only the prediction column, so proba mirrors pred (matches predict_proba).
        if pred_proba is None:
            pred_proba = pred
        if include_predict:
            return pred, pred_proba
        return pred_proba
