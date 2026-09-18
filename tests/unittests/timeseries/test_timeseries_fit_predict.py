"""Pure-unit tests for TimeSeriesCloudPredictor.fit_predict wiring."""

from unittest import mock

import pandas as pd

from autogluon.cloud import TimeSeriesCloudPredictor


def test_fit_predict_keeps_predictor_upload_default():
    with mock.patch.object(TimeSeriesCloudPredictor, "__init__", lambda self: None):
        predictor = TimeSeriesCloudPredictor()
    predictor.fit = mock.MagicMock()
    predictor.backend = mock.MagicMock()
    expected = pd.DataFrame({"item_id": ["A"], "timestamp": ["2024-01-01"], "mean": [1.0]})
    predictor.backend.get_fit_predict_results.return_value = expected

    result = predictor.fit_predict(
        train_data="train.csv",
        predictor_init_args={"prediction_length": 1},
    )

    extra_ag_args = predictor.fit.call_args.kwargs["backend_kwargs"]["extra_ag_args"]
    assert extra_ag_args == {"predict_after_fit": True}
    pd.testing.assert_frame_equal(result, expected)
