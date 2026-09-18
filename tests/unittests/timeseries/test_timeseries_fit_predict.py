"""Pure-unit tests for TimeSeriesCloudPredictor.fit_predict wiring."""

from unittest import mock

import pandas as pd

from autogluon.cloud import TimeSeriesCloudPredictor


def test_fit_predict_skips_predictor_upload_and_preserves_extra_ag_args():
    with mock.patch.object(TimeSeriesCloudPredictor, "__init__", lambda self: None):
        predictor = TimeSeriesCloudPredictor()
    predictor.fit = mock.MagicMock()
    predictor.backend = mock.MagicMock()
    expected = pd.DataFrame({"item_id": ["A"], "timestamp": ["2024-01-01"], "mean": [1.0]})
    predictor.backend.get_fit_predict_results.return_value = expected

    result = predictor.fit_predict(
        train_data="train.csv",
        predictor_init_args={"prediction_length": 1},
        backend_kwargs={"extra_ag_args": {"custom_arg": "value"}},
    )

    extra_ag_args = predictor.fit.call_args.kwargs["backend_kwargs"]["extra_ag_args"]
    assert extra_ag_args == {
        "custom_arg": "value",
        "predict_after_fit": True,
        "skip_predictor_upload": True,
    }
    pd.testing.assert_frame_equal(result, expected)
