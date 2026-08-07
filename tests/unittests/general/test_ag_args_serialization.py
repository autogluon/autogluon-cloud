"""Unit tests for the JSON serialization of remote-training args (see backend.dumps_ag_args)."""

import json

import pytest

from autogluon.cloud.backend.backend import dumps_ag_args


def test_when_config_is_json_native_then_roundtrips():
    config = {
        "predictor_type": "tabular",
        "predictor_init_args": {"label": "y", "eval_metric": "roc_auc"},
        "predictor_fit_args": {"presets": "best_quality", "time_limit": 60},
        "leaderboard": True,
    }
    assert json.loads(dumps_ag_args(config)) == config


def test_when_init_arg_not_serializable_then_error_names_that_arg():
    config = {
        "predictor_init_args": {"label": "y", "eval_metric": object()},
        "predictor_fit_args": {},
    }
    with pytest.raises(TypeError) as exc:
        dumps_ag_args(config)
    assert "`eval_metric`" in str(exc.value)


def test_when_fit_arg_not_serializable_then_error_names_that_arg():
    config = {
        "predictor_init_args": {"label": "y"},
        "predictor_fit_args": {"presets": "best_quality", "hyperparameters": {"GBM": object}},
    }
    with pytest.raises(TypeError) as exc:
        dumps_ag_args(config)
    assert "`hyperparameters`" in str(exc.value)


def test_error_message_does_not_leak_internal_ag_args_name():
    config = {"predictor_init_args": {"eval_metric": object()}, "predictor_fit_args": {}}
    with pytest.raises(TypeError) as exc:
        dumps_ag_args(config)
    assert "ag_args" not in str(exc.value)
