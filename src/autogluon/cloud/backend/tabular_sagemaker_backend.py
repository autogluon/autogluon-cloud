import logging
import os
from typing import Any, Dict, Optional, Union

import pandas as pd

from autogluon.common.loaders import load_pd

from .constant import TABULAR_SAGEMAKER
from .sagemaker_backend import SagemakerBackend

logger = logging.getLogger(__name__)


class TabularSagemakerBackend(SagemakerBackend):
    name = TABULAR_SAGEMAKER

    def parse_backend_fit_kwargs(self, kwargs: Dict) -> Dict[str, Any]:
        parsed = super().parse_backend_fit_kwargs(kwargs)
        parsed["test_data"] = kwargs.get("test_data", None)
        return parsed

    def fit(
        self,
        *,
        predictor_init_args: Dict[str, Any],
        predictor_fit_args: Dict[str, Any],
        data_channels: Dict[str, Optional[Union[str, pd.DataFrame]]],
        test_data: Optional[Union[str, pd.DataFrame]] = None,
        extra_ag_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Fit a TabularPredictor in SageMaker.

        ``test_data`` (if present in ``data_channels`` or passed explicitly) is only honored when
        ``extra_ag_args["predict_after_fit"]`` is True, in which case the training container also runs
        in-job prediction on it (the ``fit_predict`` path).
        """
        extra_ag_args = dict(extra_ag_args or {})
        if test_data is not None:
            if not extra_ag_args.get("predict_after_fit", False):
                raise ValueError("`test_data` should only be provided if `predict_after_fit=True`.")
            data_channels = dict(data_channels)
            data_channels["test_data"] = self._validate_test_data(
                train_data=data_channels["train_data"],
                test_data=test_data,
                predictor_init_args=predictor_init_args,
            )

        super().fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            data_channels=data_channels,
            extra_ag_args=extra_ag_args,
            **kwargs,
        )

    def _validate_test_data(
        self,
        *,
        train_data: Union[str, pd.DataFrame],
        test_data: Union[str, pd.DataFrame],
        predictor_init_args: Dict[str, Any],
    ) -> pd.DataFrame:
        """Validate ``test_data`` client-side before launching the SageMaker job.

        A fused fit_predict job means a schema typo wastes the whole training run, so we check up front that
        ``test_data`` covers every training feature column (the train columns minus the label). Returns the
        (possibly loaded) ``test_data`` DataFrame so the caller can reuse it as a data channel.
        """
        if isinstance(train_data, (str, os.PathLike)):
            train_data = load_pd.load(str(train_data))
        if isinstance(test_data, (str, os.PathLike)):
            test_data = load_pd.load(str(test_data))

        label = predictor_init_args.get("label")
        feature_columns = [c for c in train_data.columns if c != label]
        missing_columns = [c for c in feature_columns if c not in test_data.columns]
        if missing_columns:
            raise ValueError(f"`test_data` is missing feature columns present in `train_data`: {missing_columns}.")

        return test_data
