import copy
from typing import Any, Dict, Optional, Union

import pandas as pd

from autogluon.common.loaders import load_pd

from ..utils.serializers import AutoGluonSerializationWrapper
from .constant import TIMESERIES_SAGEMAKER
from .sagemaker_backend import SagemakerBackend


class TimeSeriesSagemakerBackend(SagemakerBackend):
    name = TIMESERIES_SAGEMAKER

    def fit(
        self,
        *,
        predictor_init_args: Dict[str, Any],
        predictor_fit_args: Dict[str, Any],
        id_column: str,
        timestamp_column: str,
        static_features: Optional[Union[str, pd.DataFrame]] = None,
        framework_version: str = "latest",
        job_name: Optional[str] = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        volume_size: int = 100,
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        autogluon_sagemaker_estimator_kwargs: Optional[Dict] = None,
        fit_kwargs: Optional[Dict] = None,
        predict_after_fit: bool = False,
    ) -> None:
        """Fit a TimeSeriesPredictor in SageMaker.

        ``id_column`` / ``timestamp_column`` are forwarded to the training script via ``ag_args.pkl``
        (see ``extra_ag_args`` on the parent ``fit``). ``static_features`` is uploaded as its own SageMaker
        channel. ``known_covariates`` (if present in ``predictor_fit_args``) is also uploaded as a separate
        channel and only honored when ``predict_after_fit=True``.
        """
        predictor_fit_args = copy.deepcopy(predictor_fit_args)
        train_data = predictor_fit_args.pop("train_data")
        tuning_data = predictor_fit_args.pop("tuning_data", None)
        known_covariates = predictor_fit_args.pop("known_covariates", None)

        if isinstance(train_data, str):
            train_data = load_pd.load(train_data)
        if isinstance(tuning_data, str):
            tuning_data = load_pd.load(tuning_data)
        if isinstance(known_covariates, str):
            known_covariates = load_pd.load(known_covariates)
        if isinstance(static_features, str):
            static_features = load_pd.load(static_features)

        if known_covariates is not None and not predict_after_fit:
            raise ValueError("`known_covariates` should only be provided if `predict_after_fit=True`.")

        predictor_fit_args["train_data"] = train_data
        predictor_fit_args["tuning_data"] = tuning_data

        extra_ag_args: Dict[str, Any] = {
            "id_column": id_column,
            "timestamp_column": timestamp_column,
            "predict_after_fit": predict_after_fit,
        }

        super().fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            framework_version=framework_version,
            job_name=job_name,
            instance_type=instance_type,
            instance_count=instance_count,
            volume_size=volume_size,
            custom_image_uri=custom_image_uri,
            wait=wait,
            autogluon_sagemaker_estimator_kwargs=autogluon_sagemaker_estimator_kwargs,
            fit_kwargs=fit_kwargs,
            known_covariates=known_covariates,
            static_features=static_features,
            extra_ag_args=extra_ag_args,
        )

    def predict_real_time(
        self,
        test_data: Union[str, pd.DataFrame],
        id_column: str,
        timestamp_column: str,
        static_features: Optional[Union[str, pd.DataFrame]] = None,
        accept: str = "application/x-parquet",
        inference_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Run a low-latency prediction against the deployed SageMaker endpoint.

        Sends ``test_data`` (and optional ``static_features`` / ``known_covariates``) as a
        single ``application/x-autogluon`` payload. ``id_column`` / ``timestamp_column`` are
        embedded in ``inference_kwargs`` so the serve script can build a TimeSeriesDataFrame.

        For datasets larger than the SageMaker 5MB endpoint limit, use ``predict()`` instead.
        """
        self._validate_predict_real_time_args(accept)

        if isinstance(test_data, str):
            test_data = load_pd.load(test_data)
        if isinstance(static_features, str):
            static_features = load_pd.load(static_features)

        if inference_kwargs is None:
            inference_kwargs = {}
        inference_kwargs["id_column"] = id_column
        inference_kwargs["timestamp_column"] = timestamp_column

        known_covariates = inference_kwargs.pop("known_covariates", None)
        if isinstance(known_covariates, str):
            known_covariates = load_pd.load(known_covariates)

        wrapper = AutoGluonSerializationWrapper(
            data=test_data,
            inference_kwargs=inference_kwargs,
            static_features=static_features,
            known_covariates=known_covariates,
        )
        pred, _ = self._predict_real_time(test_data=wrapper, accept=accept, split_pred_proba=False)
        return pred

    def predict_proba_real_time(self, **kwargs) -> pd.DataFrame:
        raise ValueError(f"{self.__class__.__name__} does not support predict_proba operation.")

    def predict(
        self,
        test_data: Union[str, pd.DataFrame],
        id_column: str,
        timestamp_column: str,
        static_features: Optional[Union[str, pd.DataFrame]] = None,
        known_covariates: Optional[Union[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Predict using SageMaker batch transform.

        Batch transform sends a single CSV body without per-record metadata, so the serve script
        relies on positional column inference: ``id_column`` at position 0, ``timestamp_column`` at
        position 1. We reorder ``test_data`` here to match that contract.

        ``static_features`` and ``known_covariates`` are not supported in this code path — SageMaker
        batch transform splits the request body and we have no second channel to attach them to. Use
        ``predict_real_time()`` instead.
        """
        if static_features is not None:
            raise NotImplementedError(
                "`static_features` is not supported for batch prediction. Use `predict_real_time()` instead."
            )
        if known_covariates is not None:
            raise NotImplementedError(
                "`known_covariates` is not supported for batch prediction. Use `predict_real_time()` instead."
            )
        if isinstance(test_data, str):
            test_data = load_pd.load(test_data)
        for required in (id_column, timestamp_column):
            if required not in test_data.columns:
                raise ValueError(f"`test_data` must contain column '{required}'.")
        # Reorder so the serve script's positional inference picks up the right columns.
        cols = test_data.columns.to_list()
        reordered = [id_column, timestamp_column] + [c for c in cols if c not in (id_column, timestamp_column)]
        test_data = test_data[reordered]
        pred, _ = super()._predict(
            test_data=test_data,
            split_pred_proba=False,
            **kwargs,
        )
        return pred

    def predict_proba(self, **kwargs) -> Optional[pd.DataFrame]:
        raise ValueError(f"{self.__class__.__name__} does not support predict_proba operation.")
