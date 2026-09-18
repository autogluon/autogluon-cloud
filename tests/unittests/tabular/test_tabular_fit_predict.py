"""Pure-unit tests for the ``fit_predict`` / ``fit_predict_proba`` wiring on TabularCloudPredictor.

These mock out the SageMaker backend entirely (no AWS) and assert the argument plumbing, return-shape
selection, and client-side validation.
"""

from unittest import mock

import pandas as pd
import pytest

from autogluon.cloud import TabularCloudPredictor
from autogluon.cloud.backend.sagemaker_backend import SagemakerBackend
from autogluon.cloud.backend.tabular_sagemaker_backend import TabularSagemakerBackend

# Pred/proba frame as written by the training container: first column is the prediction,
# remaining columns are `<class>_proba`.
CLASSIFICATION_FRAME = pd.DataFrame({"class": ["a", "b"], "a_proba": [0.7, 0.3], "b_proba": [0.3, 0.7]})
REGRESSION_FRAME = pd.DataFrame({"target": [1.5, 2.5]})

DEFAULT_ARGS = dict(train_data="train.csv", test_data="test.csv", predictor_init_args={"label": "class"})


@pytest.fixture
def cloud_predictor():
    """A TabularCloudPredictor with `fit` and the backend mocked out — no AWS interaction."""
    with mock.patch.object(TabularCloudPredictor, "__init__", lambda self: None):
        predictor = TabularCloudPredictor()
    predictor.fit = mock.MagicMock()
    predictor.backend = mock.MagicMock()
    predictor.backend.get_fit_predict_results.return_value = CLASSIFICATION_FRAME
    return predictor


def test_when_fit_predict_then_launches_predict_job_and_returns_prediction_series(cloud_predictor):
    pred = cloud_predictor.fit_predict(**DEFAULT_ARGS)
    extra_ag_args = cloud_predictor.fit.call_args.kwargs["backend_kwargs"]["extra_ag_args"]

    assert extra_ag_args["predict_after_fit"] is True
    assert "predictions_path" not in extra_ag_args  # not passed -> backend fills in a default
    assert isinstance(pred, pd.Series)
    assert pred.tolist() == ["a", "b"]


def test_when_predictions_path_given_then_forwarded_to_backend(cloud_predictor):
    cloud_predictor.fit_predict(**DEFAULT_ARGS, predictions_path="s3://bucket/key/predictions.csv")
    extra_ag_args = cloud_predictor.fit.call_args.kwargs["backend_kwargs"]["extra_ag_args"]
    assert extra_ag_args["predictions_path"] == "s3://bucket/key/predictions.csv"


def test_when_fit_predict_proba_then_returns_prediction_and_flat_proba_columns(cloud_predictor):
    pred, proba = cloud_predictor.fit_predict_proba(**DEFAULT_ARGS, include_predict=True)
    assert pred.tolist() == ["a", "b"]
    # Columns must be flat class labels (matching predict_proba), not the `_proba`-suffixed or MultiIndex
    # form produced by the training container.
    assert proba.columns.tolist() == ["a", "b"]
    assert proba["a"].tolist() == [0.7, 0.3]


def test_when_include_predict_false_then_returns_only_proba(cloud_predictor):
    proba = cloud_predictor.fit_predict_proba(**DEFAULT_ARGS, include_predict=False)
    assert isinstance(proba, pd.DataFrame)
    assert proba.columns.tolist() == ["a", "b"]


def test_when_regression_then_proba_equals_pred(cloud_predictor):
    cloud_predictor.backend.get_fit_predict_results.return_value = REGRESSION_FRAME
    pred, proba = cloud_predictor.fit_predict_proba(**{**DEFAULT_ARGS, "predictor_init_args": {"label": "target"}})
    assert pred.tolist() == [1.5, 2.5]
    pd.testing.assert_series_equal(pred, proba)


def test_when_wait_false_then_returns_none_without_fetching_results(cloud_predictor):
    assert cloud_predictor.fit_predict(**DEFAULT_ARGS, wait=False) is None
    assert cloud_predictor.fit_predict_proba(**DEFAULT_ARGS, wait=False) is None
    cloud_predictor.backend.get_fit_predict_results.assert_not_called()


# ----------------------------------------------------------------- backend-level validation


@pytest.fixture
def backend():
    """A TabularSagemakerBackend with AWS init and the base `fit` stubbed out."""
    with mock.patch.object(TabularSagemakerBackend, "__init__", lambda self: None):
        backend = TabularSagemakerBackend()
    with mock.patch.object(SagemakerBackend, "fit"):
        yield backend


def _backend_fit(backend, train, test=None, label="y"):
    data_channels = {"train_data": train} if test is None else {"train_data": train, "test_data": test}
    backend.fit(predictor_init_args={"label": label}, predictor_fit_args={}, data_channels=data_channels)


def test_when_test_data_omits_label_column_then_validation_passes(backend):
    train = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    test = pd.DataFrame({"x": [5, 6]})  # no label column — only feature columns are required
    _backend_fit(backend, train, test)  # does not raise


def test_when_test_data_missing_feature_columns_then_raises(backend):
    train = pd.DataFrame({"x": [1, 2], "z": [3, 4], "y": [0, 1]})
    test = pd.DataFrame({"x": [5]})  # missing feature column `z`
    with pytest.raises(ValueError, match="missing feature columns"):
        _backend_fit(backend, train, test)


def test_when_label_missing_from_init_args_then_raises(backend):
    train = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    with pytest.raises(ValueError, match="must contain `label`"):
        backend.fit(predictor_init_args={}, predictor_fit_args={}, data_channels={"train_data": train})


def test_when_label_absent_from_train_data_then_raises(backend):
    train = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    with pytest.raises(ValueError, match="not present in `train_data`"):
        _backend_fit(backend, train, label="nonexistent")
