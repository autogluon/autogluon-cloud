import os
from typing import Any, Dict, Optional, Union

import pandas as pd

from autogluon.common.loaders import load_pd

from .constant import TABULAR_SAGEMAKER
from .sagemaker_backend import SagemakerBackend


class TabularSagemakerBackend(SagemakerBackend):
    name = TABULAR_SAGEMAKER

    def fit(
        self,
        *,
        predictor_init_args: Dict[str, Any],
        predictor_fit_args: Dict[str, Any],
        data_channels: Dict[str, Optional[Union[str, pd.DataFrame]]],
        **kwargs,
    ) -> None:
        data_channels = self._validate_data_channels(
            data_channels=data_channels,
            predictor_init_args=predictor_init_args,
        )
        super().fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            data_channels=data_channels,
            **kwargs,
        )

    def _validate_data_channels(
        self,
        *,
        data_channels: Dict[str, Optional[Union[str, pd.DataFrame]]],
        predictor_init_args: Dict[str, Any],
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Validate tabular data channels client-side before launching the SageMaker job.

        Resolves path inputs via ``load_pd.load`` so column checks can run, and returns the loaded channel
        dict for the caller to reuse.

        A fused fit_predict job means a schema typo wastes the whole training run, so we check up front that
        ``test_data`` (if present) covers every training feature column (the train columns minus the label).
        """
        loaded: Dict[str, Optional[pd.DataFrame]] = {}
        for name, df in data_channels.items():
            if isinstance(df, (str, os.PathLike)):
                df = load_pd.load(str(df))
            loaded[name] = df

        if "label" not in predictor_init_args:
            raise ValueError("`predictor_init_args` must contain `label` for a Tabular predictor.")
        label = predictor_init_args["label"]
        if label not in loaded["train_data"].columns:
            raise ValueError(f"Label column {label!r} is not present in `train_data`.")

        test_data = loaded.get("test_data")
        if test_data is not None:
            feature_columns = [c for c in loaded["train_data"].columns if c != label]
            missing_columns = [c for c in feature_columns if c not in test_data.columns]
            if missing_columns:
                raise ValueError(f"`test_data` is missing feature columns present in `train_data`: {missing_columns}.")

        return loaded
