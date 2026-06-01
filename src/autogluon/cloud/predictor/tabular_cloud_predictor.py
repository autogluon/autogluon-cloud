from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from ..backend.constant import RAY_AWS, SAGEMAKER, TABULAR_RAY_AWS, TABULAR_SAGEMAKER
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
        framework_version: str = "latest",
        job_name: Optional[str] = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        volume_size: int = 256,
        custom_image_uri: Optional[str] = None,
        timeout: int = 24 * 60 * 60,
        wait: bool = True,
        predictions_path: Optional[str] = None,
        backend_kwargs: Optional[Dict] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fit and predict in a single SageMaker training job.

        This is useful for tabular foundation-model workflows (e.g. Mitra) where "fit" is essentially
        loading a pretrained model. Running fit and predict in the same job avoids the SageMaker
        startup overhead twice.

        For classification tasks, the returned DataFrame matches the output of
        :meth:`TabularCloudPredictor.predict_proba` with ``include_predict=True`` — the first
        column is the predicted class and the remaining columns are class probabilities (suffixed
        ``_proba``). Use :func:`autogluon.cloud.utils.utils.split_pred_and_pred_proba` to split.

        Parameters
        ----------
        train_data: Union[str, pathlib.Path, pd.DataFrame]
            Labeled training data, as a DataFrame or local/S3 path to a data file.
        test_data: Union[str, pathlib.Path, pd.DataFrame]
            Unlabeled data to predict on, as a DataFrame or local/S3 path to a data file.
        predictor_init_args: dict
            Arguments forwarded to ``TabularPredictor()`` (must include ``label``).
        predictor_fit_args: Optional[dict], default = None
            Additional fit args forwarded to ``TabularPredictor.fit()``.
        predictions_path: Optional[str]
            S3 URL where predictions will be written by the training container. Defaults to
            ``{cloud_output_path}/{job_name}/predictions.csv``. Must end in ``.csv`` or ``.parquet``.
        framework_version, job_name, instance_type, instance_count, volume_size, custom_image_uri,
        timeout, wait, backend_kwargs:
            Same semantics as ``fit()``.

        Returns
        -------
        Optional[pd.DataFrame]
            Predictions as a DataFrame. Returns ``None`` when ``wait`` is False.
        """
        assert not self.backend.is_fit, (
            "Predictor is already fit! To fit additional models, create a new `CloudPredictor`"
        )
        if backend_kwargs is None:
            backend_kwargs = {}
        else:
            backend_kwargs = dict(backend_kwargs)

        extra_ag_args: Dict[str, Any] = {"predict_after_fit": True}
        if predictions_path is not None:
            extra_ag_args["predictions_path"] = predictions_path
        backend_kwargs["extra_ag_args"] = extra_ag_args

        predictor_fit_args = {} if predictor_fit_args is None else dict(predictor_fit_args)
        data_channels = {
            "train_data": train_data,
            "test_data": test_data,
        }

        backend_kwargs = self.backend.parse_backend_fit_kwargs(backend_kwargs)
        self.backend.fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            data_channels=data_channels,
            framework_version=framework_version,
            job_name=job_name,
            instance_type=instance_type,
            instance_count=instance_count,
            volume_size=volume_size,
            custom_image_uri=custom_image_uri,
            timeout=timeout,
            wait=wait,
            **backend_kwargs,
        )

        if not wait:
            logger.info(
                "fit_predict job launched asynchronously. Use `get_fit_job_status()` "
                "to poll, then `get_fit_predict_results()` to fetch predictions."
            )
            return None

        return self.get_fit_predict_results()
