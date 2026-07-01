"""Pure-unit tests for the ``fit_predict`` / ``fit_predict_proba`` wiring on TabularCloudPredictor.

These mock out the SageMaker backend entirely (no AWS) and assert the argument plumbing, return-shape
selection, and client-side validation.
"""

from unittest import mock

import pandas as pd
import pytest

from autogluon.cloud import TabularCloudPredictor
from autogluon.cloud.backend.tabular_sagemaker_backend import TabularSagemakerBackend


@pytest.fixture
def cloud_predictor():
    """A TabularCloudPredictor whose backend is a MagicMock — no AWS interaction."""
    with mock.patch.object(TabularCloudPredictor, "__init__", lambda self: None):
        predictor = TabularCloudPredictor()
    predictor.backend = mock.MagicMock()
    return predictor


# Pred/proba frame as written by the training container: first column is the prediction,
# remaining columns are `<class>_proba`.
def _classification_frame():
    return pd.DataFrame(
        {
            "class": ["a", "b"],
            "a_proba": [0.7, 0.3],
            "b_proba": [0.3, 0.7],
        }
    )


def _regression_frame():
    return pd.DataFrame({"target": [1.5, 2.5]})


def test_fit_predict_passes_predict_after_fit_and_test_data(cloud_predictor):
    cloud_predictor.backend.get_fit_predict_results.return_value = _classification_frame()
    test_data = pd.DataFrame({"x": [1, 2]})

    with mock.patch.object(cloud_predictor, "fit") as fit:
        cloud_predictor.fit_predict(
            train_data="train.csv",
            test_data=test_data,
            predictor_init_args={"label": "class"},
        )

    backend_kwargs = fit.call_args.kwargs["backend_kwargs"]
    assert backend_kwargs["extra_ag_args"]["predict_after_fit"] is True
    assert fit.call_args.kwargs["test_data"] is test_data
    assert "predictions_path" not in backend_kwargs["extra_ag_args"]


def test_fit_predict_forwards_predictions_path(cloud_predictor):
    cloud_predictor.backend.get_fit_predict_results.return_value = _classification_frame()

    with mock.patch.object(cloud_predictor, "fit") as fit:
        cloud_predictor.fit_predict(
            train_data="train.csv",
            test_data="test.csv",
            predictor_init_args={"label": "class"},
            predictions_path="s3://bucket/key/predictions.csv",
        )

    extra_ag_args = fit.call_args.kwargs["backend_kwargs"]["extra_ag_args"]
    assert extra_ag_args["predictions_path"] == "s3://bucket/key/predictions.csv"


def test_fit_predict_returns_prediction_series(cloud_predictor):
    cloud_predictor.backend.get_fit_predict_results.return_value = _classification_frame()

    with mock.patch.object(cloud_predictor, "fit"):
        pred = cloud_predictor.fit_predict(
            train_data="train.csv",
            test_data="test.csv",
            predictor_init_args={"label": "class"},
        )

    assert isinstance(pred, pd.Series)
    assert pred.tolist() == ["a", "b"]


def test_fit_predict_wait_false_returns_none(cloud_predictor):
    with mock.patch.object(cloud_predictor, "fit"):
        result = cloud_predictor.fit_predict(
            train_data="train.csv",
            test_data="test.csv",
            predictor_init_args={"label": "class"},
            wait=False,
        )

    assert result is None
    cloud_predictor.backend.get_fit_predict_results.assert_not_called()


def test_fit_predict_proba_include_predict_true_returns_pred_and_proba(cloud_predictor):
    cloud_predictor.backend.get_fit_predict_results.return_value = _classification_frame()

    with mock.patch.object(cloud_predictor, "fit"):
        result = cloud_predictor.fit_predict_proba(
            train_data="train.csv",
            test_data="test.csv",
            predictor_init_args={"label": "class"},
            include_predict=True,
        )

    assert isinstance(result, tuple)
    pred, proba = result
    assert pred.tolist() == ["a", "b"]
    assert isinstance(proba, pd.DataFrame)
    assert proba.shape == (2, 2)


def test_fit_predict_proba_include_predict_false_returns_only_proba(cloud_predictor):
    cloud_predictor.backend.get_fit_predict_results.return_value = _classification_frame()

    with mock.patch.object(cloud_predictor, "fit"):
        result = cloud_predictor.fit_predict_proba(
            train_data="train.csv",
            test_data="test.csv",
            predictor_init_args={"label": "class"},
            include_predict=False,
        )

    assert isinstance(result, pd.DataFrame)


def test_fit_predict_proba_regression_proba_equals_pred(cloud_predictor):
    cloud_predictor.backend.get_fit_predict_results.return_value = _regression_frame()

    with mock.patch.object(cloud_predictor, "fit"):
        pred, proba = cloud_predictor.fit_predict_proba(
            train_data="train.csv",
            test_data="test.csv",
            predictor_init_args={"label": "target"},
        )

    assert pred.tolist() == [1.5, 2.5]
    pd.testing.assert_series_equal(pred, proba)


def test_fit_predict_proba_wait_false_returns_none(cloud_predictor):
    with mock.patch.object(cloud_predictor, "fit"):
        result = cloud_predictor.fit_predict_proba(
            train_data="train.csv",
            test_data="test.csv",
            predictor_init_args={"label": "class"},
            wait=False,
        )

    assert result is None
    cloud_predictor.backend.get_fit_predict_results.assert_not_called()


# ----------------------------------------------------------------- backend-level validation


@pytest.fixture
def backend():
    """A TabularSagemakerBackend with the base ``fit`` and AWS init stubbed out."""
    with mock.patch.object(TabularSagemakerBackend, "__init__", lambda self: None):
        backend = TabularSagemakerBackend()
    return backend


def test_backend_fit_rejects_test_data_missing_feature_columns(backend):
    train = pd.DataFrame({"x": [1, 2], "z": [3, 4], "y": [0, 1]})
    test = pd.DataFrame({"x": [5]})  # missing feature column `z`
    with pytest.raises(ValueError, match="missing feature columns"):
        backend.fit(
            predictor_init_args={"label": "y"},
            predictor_fit_args={},
            data_channels={"train_data": train, "test_data": test},
        )


def test_backend_fit_forwards_test_data_channel(backend):
    train = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    test = pd.DataFrame({"x": [5, 6]})

    with mock.patch.object(TabularSagemakerBackend.__bases__[0], "fit") as super_fit:
        backend.fit(
            predictor_init_args={"label": "y"},
            predictor_fit_args={},
            data_channels={"train_data": train, "test_data": test},
        )

    forwarded_channels = super_fit.call_args.kwargs["data_channels"]
    assert "test_data" in forwarded_channels
    pd.testing.assert_frame_equal(forwarded_channels["test_data"], test)


def test_backend_fit_test_data_allows_label_column_absent(backend):
    """test_data need not contain the label column, only the feature columns."""
    train = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    test = pd.DataFrame({"x": [5, 6]})  # no `y`, that's fine

    with mock.patch.object(TabularSagemakerBackend.__bases__[0], "fit"):
        backend.fit(
            predictor_init_args={"label": "y"},
            predictor_fit_args={},
            data_channels={"train_data": train, "test_data": test},
        )
