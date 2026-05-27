import logging
import os
from typing import Any, Dict, Optional, Union

import pandas as pd

from autogluon.common.loaders import load_pd

from ..utils.serializers import AutoGluonSerializationWrapper, AutoGluonSerializer
from .constant import TIMESERIES_SAGEMAKER
from .sagemaker_backend import SagemakerBackend

logger = logging.getLogger(__name__)


class TimeSeriesSagemakerBackend(SagemakerBackend):
    name = TIMESERIES_SAGEMAKER

    def fit(
        self,
        *,
        predictor_init_args: Dict[str, Any],
        predictor_fit_args: Dict[str, Any],
        data_channels: Dict[str, Optional[Union[str, pd.DataFrame]]],
        id_column: str,
        timestamp_column: str,
        framework_version: str = "latest",
        job_name: Optional[str] = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        volume_size: int = 100,
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        autogluon_sagemaker_estimator_kwargs: Optional[Dict] = None,
        fit_kwargs: Optional[Dict] = None,
        extra_ag_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fit a TimeSeriesPredictor in SageMaker.

        ``id_column`` / ``timestamp_column`` are forwarded to the training script via ``ag_args.pkl``.
        ``known_covariates`` (if present in ``data_channels``) is only honored when
        ``extra_ag_args["predict_after_fit"]`` is True.
        """
        extra_ag_args = {**(extra_ag_args or {}), "id_column": id_column, "timestamp_column": timestamp_column}
        if data_channels.get("known_covariates") is not None and not extra_ag_args.get("predict_after_fit", False):
            raise ValueError("`known_covariates` should only be provided if `predict_after_fit=True`.")
        data_channels = self._validate_data_channels(
            data_channels=data_channels,
            predictor_init_args=predictor_init_args,
            id_column=id_column,
            timestamp_column=timestamp_column,
        )

        super().fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            data_channels=data_channels,
            framework_version=framework_version,
            job_name=job_name,
            instance_type=instance_type,
            instance_count=instance_count,
            volume_size=volume_size,
            custom_image_uri=custom_image_uri,
            wait=wait,
            autogluon_sagemaker_estimator_kwargs=autogluon_sagemaker_estimator_kwargs,
            fit_kwargs=fit_kwargs,
            extra_ag_args=extra_ag_args,
        )

    def predict_real_time(
        self,
        test_data: Union[str, pd.DataFrame],
        static_features: Optional[Union[str, pd.DataFrame]] = None,
        known_covariates: Optional[Union[str, pd.DataFrame]] = None,
        accept: str = "application/x-parquet",
        inference_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Predict with the deployed SageMaker endpoint. A deployed SageMaker endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict()` instead.

        Parameters
        ----------
        test_data: Union(str, pandas.DataFrame)
            The test data to be inferenced.
            Can be a pandas.DataFrame or a local path to a csv file.
        static_features: Optional[Union[str, pd.DataFrame]]
             An optional data frame describing the metadata attributes of individual items in the item index.
             For more detail, please refer to `TimeSeriesDataFrame` documentation:
             https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesDataFrame.html
        known_covariates: Optional[Union[str, pd.DataFrame]]
            Future values of the known covariates over the forecast horizon.
        accept: str, default = application/x-parquet
            Type of accept output content.
            Valid options are application/x-parquet, text/csv, application/json
        inference_kwargs: Optional[Dict[str, Any]], default = None
            Additional args that you would pass to `predict` calls of an AutoGluon logic

        Returns
        -------
        Pandas.DataFrame
        Predict results in DataFrame
        """
        self._validate_predict_real_time_args(accept)

        if isinstance(test_data, str):
            test_data = load_pd.load(test_data)
        if isinstance(static_features, str):
            static_features = load_pd.load(static_features)
        if isinstance(known_covariates, str):
            known_covariates = load_pd.load(known_covariates)

        wrapper = AutoGluonSerializationWrapper(
            data=test_data,
            inference_kwargs=inference_kwargs or {},
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
        static_features: Optional[Union[str, pd.DataFrame]] = None,
        known_covariates: Optional[Union[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
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
        static_features: Optional[Union[str, pd.DataFrame]]
             An optional data frame describing the metadata attributes of individual items in the item index.
             For more detail, please refer to `TimeSeriesDataFrame` documentation:
             https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesDataFrame.html
        known_covariates: Optional[Union[str, pd.DataFrame]]
            Future values of the known covariates over the forecast horizon.
        kwargs:
            Refer to `SagemakerBackend.predict()`
        """
        if isinstance(test_data, str):
            test_data = load_pd.load(test_data)
        if isinstance(static_features, str):
            static_features = load_pd.load(static_features)
        if isinstance(known_covariates, str):
            known_covariates = load_pd.load(known_covariates)

        wrapper = AutoGluonSerializationWrapper(
            data=test_data,
            inference_kwargs={},
            static_features=static_features,
            known_covariates=known_covariates,
        )
        # Pickle the request body to disk and pass the path through. Force content_type / split_type so
        # SageMaker sends the whole body in one transform_fn call — preserves static_features /
        # known_covariates side channels.
        payload_dir = os.path.join(self.local_output_path, "utils")
        os.makedirs(payload_dir, exist_ok=True)
        payload_path = os.path.join(payload_dir, "predict_payload.pkl")
        with open(payload_path, "wb") as f:
            f.write(AutoGluonSerializer().serialize(wrapper))
        transform_kwargs = kwargs.pop("transform_kwargs", None) or {}
        transform_kwargs["content_type"] = "application/x-autogluon"
        transform_kwargs["split_type"] = "None"

        pred, _ = super()._predict(
            test_data=payload_path,
            split_pred_proba=False,
            transform_kwargs=transform_kwargs,
            **kwargs,
        )
        return pred

    def predict_proba(
        self,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        raise ValueError(f"{self.__class__.__name__} does not support predict_proba operation.")

    def _validate_data_channels(
        self,
        *,
        data_channels: Dict[str, Optional[Union[str, pd.DataFrame]]],
        predictor_init_args: Dict[str, Any],
        id_column: str,
        timestamp_column: str,
    ) -> Dict[str, Optional[Union[str, pd.DataFrame]]]:
        """Validate time-series data channels client-side before launching the SageMaker job.

        Resolves ``str`` paths via ``load_pd.load`` so column checks can run, and returns the (possibly loaded)
        channel dict for the caller to reuse.

        Validates:
        - ``train_data`` and ``tuning_data`` contain ``id_column``, ``timestamp_column``, and the predictor's
          ``target`` column.
        - ``known_covariates`` (if present) contains ``id_column`` / ``timestamp_column`` and every name in
          ``predictor_init_args['known_covariates_names']``. Warns about extra columns that will be ignored.
        - If ``known_covariates_names`` is set, ``train_data`` must also contain those columns.
        """
        target = predictor_init_args.get("target", "target")
        loaded: Dict[str, Optional[Union[str, pd.DataFrame]]] = {}
        for name, df in data_channels.items():
            if isinstance(df, str):
                df = load_pd.load(df)
            loaded[name] = df

        for name in ("train_data", "tuning_data"):
            df = loaded.get(name)
            if df is None:
                continue
            for required in (id_column, timestamp_column, target):
                if required not in df.columns:
                    raise ValueError(f"`{name}` must contain column '{required}'.")

        known_covariates = loaded.get("known_covariates")
        names = predictor_init_args.get("known_covariates_names")
        if known_covariates is not None:
            if names is not None and (isinstance(names, str) or not isinstance(names, (list, tuple))):
                raise ValueError(
                    "`predictor_init_args['known_covariates_names']` must be a list or tuple of strings, "
                    f"got {type(names).__name__}."
                )
            kc_cols = known_covariates.columns.tolist()
            for required in (id_column, timestamp_column):
                if required not in kc_cols:
                    raise ValueError(f"`known_covariates` must contain column '{required}'.")
            if names:
                missing_in_covs = [n for n in names if n not in kc_cols]
                if missing_in_covs:
                    raise ValueError(
                        f"`known_covariates` is missing columns listed in `known_covariates_names`: {missing_in_covs}."
                    )
                extra_in_covs = [c for c in kc_cols if c not in names and c not in (id_column, timestamp_column)]
                if extra_in_covs:
                    logger.warning(
                        f"`known_covariates` has columns not listed in `known_covariates_names`: {extra_in_covs}. "
                        "These will be ignored by the predictor."
                    )

        if names:
            train_data = loaded.get("train_data")
            if isinstance(train_data, pd.DataFrame):
                missing_in_train = [n for n in names if n not in train_data.columns]
                if missing_in_train:
                    raise ValueError(
                        f"`train_data` is missing columns listed in `known_covariates_names`: {missing_in_train}."
                    )

        return loaded
