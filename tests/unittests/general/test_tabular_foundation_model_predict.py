"""Pure-unit tests for TabularFoundationModel.predict / predict_proba (no AWS).

The batch path reuses the exact mechanism as ``TabularCloudPredictor.fit_predict_proba``:
``backend.fit(extra_ag_args={"predict_after_fit": True})`` -> ``backend.get_fit_predict_results()``
-> ``split_pred_and_pred_proba``. The equivalence test pins that the two entry points return the
same thing off the same backend frame; the remaining tests cover the branches unique to the FM path.
"""

from unittest import mock

import pandas as pd
import pytest

from autogluon.cloud import TabularCloudPredictor
from autogluon.cloud.endpoint.prediction_future import JobPredictionFuture
from autogluon.cloud.model import FoundationModel

# Pred/proba frame as written by the training container: first column is the prediction,
# remaining columns are `<class>_proba`.
CLASSIFICATION_FRAME = pd.DataFrame({"class": ["a", "b"], "a_proba": [0.7, 0.3], "b_proba": [0.3, 0.7]})
REGRESSION_FRAME = pd.DataFrame({"target": [1.5, 2.5]})

PREDICT_ARGS = dict(train_data="train.csv", test_data="test.csv", label="class")


@pytest.fixture(autouse=True)
def _stub_aws(monkeypatch):
    """Avoid touching AWS / config files during FoundationModel construction."""
    monkeypatch.setattr(
        "autogluon.cloud.model.foundation_model.resolve_cloud_output_path",
        lambda path, backend_name: path or "s3://stub/output",
    )
    monkeypatch.setattr(
        "autogluon.cloud.backend.backend_factory.BackendFactory.get_backend",
        lambda **kwargs: mock.MagicMock(role_arn="arn:aws:iam::0:role/stub"),
    )


def _make_fm(model_id="mitra-classifier", result=CLASSIFICATION_FRAME):
    fm = FoundationModel(model_id, cloud_output_path="s3://b")
    fm._backend.get_fit_predict_results.return_value = result
    return fm


def test_predict_returns_prediction_series():
    fm = _make_fm()
    pred = fm.predict(**PREDICT_ARGS)
    assert isinstance(pred, pd.Series)
    assert pred.tolist() == ["a", "b"]


def test_predict_launches_predict_after_fit_job():
    fm = _make_fm()
    fm.predict(**PREDICT_ARGS)
    extra_ag_args = fm._backend.fit.call_args.kwargs["extra_ag_args"]
    assert extra_ag_args["predict_after_fit"] is True
    assert "predictions_path" not in extra_ag_args  # not passed -> backend fills in a default


def test_predict_proba_returns_prediction_and_flat_proba_columns():
    fm = _make_fm()
    pred, proba = fm.predict_proba(**PREDICT_ARGS, include_predict=True)
    assert pred.tolist() == ["a", "b"]
    # Columns must be flat class labels (matching predict_proba), not the `_proba`-suffixed form
    # produced by the training container.
    assert proba.columns.tolist() == ["a", "b"]
    assert proba["a"].tolist() == [0.7, 0.3]


def test_predict_proba_include_predict_false_returns_only_proba():
    fm = _make_fm()
    proba = fm.predict_proba(**PREDICT_ARGS, include_predict=False)
    assert isinstance(proba, pd.DataFrame)
    assert proba.columns.tolist() == ["a", "b"]


def test_regression_proba_equals_pred():
    fm = _make_fm(model_id="mitra-regressor", result=REGRESSION_FRAME)
    pred, proba = fm.predict_proba(train_data="t.csv", test_data="s.csv", label="target")
    assert pred.tolist() == [1.5, 2.5]
    pd.testing.assert_series_equal(pred, proba)


def test_predictions_path_forwarded_to_backend():
    fm = _make_fm()
    fm.predict(**PREDICT_ARGS, predictions_path="s3://bucket/key/predictions.csv")
    extra_ag_args = fm._backend.fit.call_args.kwargs["extra_ag_args"]
    assert extra_ag_args["predictions_path"] == "s3://bucket/key/predictions.csv"


def test_wait_false_returns_future_without_fetching():
    fm = _make_fm()
    future = fm.predict(**PREDICT_ARGS, wait=False)
    assert isinstance(future, JobPredictionFuture)
    fm._backend.get_fit_predict_results.assert_not_called()


# ----------------------------------------------------------------- equivalence with TabularCloudPredictor


def _make_tcp(result=CLASSIFICATION_FRAME):
    """A TabularCloudPredictor with `fit` and the backend mocked out — no AWS interaction."""
    with mock.patch.object(TabularCloudPredictor, "__init__", lambda self: None):
        tcp = TabularCloudPredictor()
    tcp.fit = mock.MagicMock()
    tcp.backend = mock.MagicMock()
    tcp.backend.get_fit_predict_results.return_value = result
    return tcp


@pytest.mark.parametrize("frame", [CLASSIFICATION_FRAME, REGRESSION_FRAME])
def test_fm_predict_matches_tcp_fit_predict(frame):
    """TabularFoundationModel.predict_proba and TabularCloudPredictor.fit_predict_proba must return the
    same (pred, proba) off the same backend frame — they wrap one shared result-loading mechanism."""
    label = "class" if frame is CLASSIFICATION_FRAME else "target"

    fm = _make_fm(model_id="mitra-classifier", result=frame)
    fm_pred, fm_proba = fm.predict_proba(train_data="t.csv", test_data="s.csv", label=label)

    tcp = _make_tcp(result=frame)
    tcp_pred, tcp_proba = tcp.fit_predict_proba(
        train_data="t.csv", test_data="s.csv", predictor_init_args={"label": label}
    )

    pd.testing.assert_series_equal(fm_pred, tcp_pred)
    if isinstance(tcp_proba, pd.DataFrame):
        pd.testing.assert_frame_equal(fm_proba, tcp_proba)
    else:
        pd.testing.assert_series_equal(fm_proba, tcp_proba)
