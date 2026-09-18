import importlib.util
import json
import sys
import types
from io import BytesIO
from pathlib import Path

import pandas as pd

from autogluon.cloud.utils.serializers import AutoGluonSerializationWrapper, AutoGluonSerializer

SERVE_SCRIPT = (
    Path(__file__).parents[3]
    / "src"
    / "autogluon"
    / "cloud"
    / "scripts"
    / "sagemaker_scripts"
    / "tabular_fm_serve.py"
)


class FakeTabularPredictor:
    instances = []
    can_predict_proba = True

    def __init__(self, *, label, problem_type, path):
        self.label = label
        self.problem_type = problem_type
        self.path = path
        self.fit_data = None
        self.fit_kwargs = None
        self.predict_data = None
        self.predict_kwargs = None
        self.__class__.instances.append(self)

    def fit(self, train_data, **kwargs):
        self.fit_data = train_data
        self.fit_kwargs = kwargs
        return self

    def predict(self, data, **kwargs):
        self.predict_data = data
        self.predict_kwargs = kwargs
        return pd.Series(["a", "b"], name=self.label)

    def predict_proba(self, data, **kwargs):
        return pd.DataFrame({"a": [0.8, 0.2], "b": [0.2, 0.8]})


def _load_serve_module(monkeypatch):
    FakeTabularPredictor.instances.clear()
    tabular_module = types.ModuleType("autogluon.tabular")
    tabular_module.TabularPredictor = FakeTabularPredictor
    monkeypatch.setitem(sys.modules, "autogluon.tabular", tabular_module)

    spec = importlib.util.spec_from_file_location("tabular_fm_serve_under_test", SERVE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_fn_downloads_remote_weights_during_startup(monkeypatch):
    config = {
        "ag_model_key": "MITRA",
        "hyperparameters": {
            "fine_tune": False,
            "hf_cls_model": "autogluon/mitra-classifier",
        },
        "problem_type": "multiclass",
    }
    monkeypatch.setenv("AG_FM_SERVE_CONFIG", json.dumps(config))
    serve = _load_serve_module(monkeypatch)
    snapshot_download = types.SimpleNamespace(calls=[])

    def _snapshot_download(**kwargs):
        snapshot_download.calls.append(kwargs)
        return "/cache/snapshot"

    monkeypatch.setattr(
        serve,
        "snapshot_download",
        _snapshot_download,
    )

    loaded_config = serve.model_fn("/opt/ml/model")

    assert loaded_config["hyperparameters"]["hf_cls_model"] == "autogluon/mitra-classifier"
    assert snapshot_download.calls == [
        {
            "repo_id": "autogluon/mitra-classifier",
            "allow_patterns": ["config.json", "model.safetensors"],
        }
    ]
    assert config["hyperparameters"]["hf_cls_model"] == "autogluon/mitra-classifier"


def test_transform_fits_request_train_data_before_predicting(monkeypatch):
    serve = _load_serve_module(monkeypatch)
    train_data = pd.DataFrame({"feature": [0, 1], "label": ["a", "b"]})
    data = pd.DataFrame({"feature": [2, 3]})
    payload = AutoGluonSerializer().serialize(
        AutoGluonSerializationWrapper(
            data=data,
            train_data=train_data,
            inference_kwargs={"label": "label"},
        )
    )
    config = {
        "ag_model_key": "MITRA",
        "hyperparameters": {"fine_tune": False, "hf_cls_model": "autogluon/mitra-classifier"},
        "problem_type": "multiclass",
    }

    body, content_type = serve.transform_fn(
        config,
        payload,
        "application/x-autogluon",
        "application/x-parquet",
    )

    predictor = FakeTabularPredictor.instances[0]
    pd.testing.assert_frame_equal(predictor.fit_data, train_data)
    pd.testing.assert_frame_equal(predictor.predict_data, data)
    assert predictor.problem_type == "multiclass"
    assert predictor.fit_kwargs == {
        "hyperparameters": {
            "MITRA": {"fine_tune": False, "hf_cls_model": "autogluon/mitra-classifier"}
        },
        "fit_weighted_ensemble": False,
    }
    assert content_type == "application/x-parquet"
    result = pd.read_parquet(BytesIO(body))
    assert result.columns.tolist() == ["label", "a_proba", "b_proba"]
    assert result["label"].tolist() == ["a", "b"]


def test_transform_requires_train_data_and_label(monkeypatch):
    serve = _load_serve_module(monkeypatch)
    data = pd.DataFrame({"feature": [2]})
    config = {"ag_model_key": "MITRA", "hyperparameters": {}, "problem_type": "multiclass"}

    payload_without_train = AutoGluonSerializer().serialize(
        AutoGluonSerializationWrapper(data=data, inference_kwargs={"label": "label"})
    )
    try:
        serve.transform_fn(config, payload_without_train, "application/x-autogluon")
    except ValueError as error:
        assert "train_data" in str(error)
    else:
        raise AssertionError("Expected missing train_data to raise")

    train_data = pd.DataFrame({"feature": [0], "label": ["a"]})
    payload_without_label = AutoGluonSerializer().serialize(
        AutoGluonSerializationWrapper(data=data, train_data=train_data, inference_kwargs={})
    )
    try:
        serve.transform_fn(config, payload_without_label, "application/x-autogluon")
    except ValueError as error:
        assert "label" in str(error)
    else:
        raise AssertionError("Expected missing label to raise")
