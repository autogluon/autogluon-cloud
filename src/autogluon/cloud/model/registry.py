"""Foundation model registry.

Maps model_id to AG-compatible configuration for deploy / predict.
"""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class FoundationModelConfig:
    problem_type: Literal["forecasting", "multiclass", "regression"]
    ag_model_key: str  # key in the AG hyperparameters dict (e.g. "Chronos", "Chronos2", "MITRA")
    model_source_uri: str  # where weights are downloaded from (e.g. "autogluon/chronos-2")
    # AG-model hyperparameter that `model_source_uri` is injected into (None => don't inject).
    model_source_hyperparameter: str | None = None
    predict_instance_type: str = "ml.m5.2xlarge"  # batch predict
    deploy_instance_type: str = "ml.g5.xlarge"  # real-time endpoint
    fit_instance_type: str = "ml.g5.xlarge"  # fine-tuning
    inference_hyperparameters: dict[str, Any] = field(default_factory=dict)  # defaults for deploy() and predict()
    training_hyperparameters: dict[str, Any] = field(default_factory=dict)  # defaults for fit()
    fine_tunable: bool = False  # whether .fit() is supported


FOUNDATION_MODEL_REGISTRY: dict[str, FoundationModelConfig] = {
    "chronos-bolt-tiny": FoundationModelConfig(
        problem_type="forecasting",
        ag_model_key="Chronos",
        model_source_uri="autogluon/chronos-bolt-tiny",
        model_source_hyperparameter="model_path",
    ),
    "chronos-bolt-small": FoundationModelConfig(
        problem_type="forecasting",
        ag_model_key="Chronos",
        model_source_uri="autogluon/chronos-bolt-small",
        model_source_hyperparameter="model_path",
    ),
    "chronos-bolt-base": FoundationModelConfig(
        problem_type="forecasting",
        ag_model_key="Chronos",
        model_source_uri="autogluon/chronos-bolt-base",
        model_source_hyperparameter="model_path",
    ),
    "chronos-2-small": FoundationModelConfig(
        problem_type="forecasting",
        ag_model_key="Chronos2",
        model_source_uri="autogluon/chronos-2-small",
        model_source_hyperparameter="model_path",
    ),
    "chronos-2": FoundationModelConfig(
        problem_type="forecasting",
        ag_model_key="Chronos2",
        model_source_uri="autogluon/chronos-2",
        model_source_hyperparameter="model_path",
    ),
    "toto-2.0-4m": FoundationModelConfig(
        problem_type="forecasting",
        ag_model_key="Toto2",
        model_source_uri="Datadog/Toto-2.0-4m",
        model_source_hyperparameter="model_path",
    ),
    "toto-2.0-22m": FoundationModelConfig(
        problem_type="forecasting",
        ag_model_key="Toto2",
        model_source_uri="Datadog/Toto-2.0-22m",
        model_source_hyperparameter="model_path",
    ),
    "toto-2.0-313m": FoundationModelConfig(
        problem_type="forecasting",
        ag_model_key="Toto2",
        model_source_uri="Datadog/Toto-2.0-313m",
        model_source_hyperparameter="model_path",
        predict_instance_type="ml.g5.xlarge",
    ),
    "toto-2.0-1b": FoundationModelConfig(
        problem_type="forecasting",
        ag_model_key="Toto2",
        model_source_uri="Datadog/Toto-2.0-1B",
        model_source_hyperparameter="model_path",
        predict_instance_type="ml.g5.xlarge",
    ),
    "toto-2.0-2.5b": FoundationModelConfig(
        problem_type="forecasting",
        ag_model_key="Toto2",
        model_source_uri="Datadog/Toto-2.0-2.5B",
        model_source_hyperparameter="model_path",
        predict_instance_type="ml.g5.xlarge",
    ),
    "mitra-classifier": FoundationModelConfig(
        problem_type="multiclass",
        ag_model_key="MITRA",
        model_source_uri="autogluon/mitra-classifier",
        model_source_hyperparameter="hf_cls_model",
        inference_hyperparameters={"fine_tune": False},
        predict_instance_type="ml.m5.4xlarge",
        deploy_instance_type="ml.m5.4xlarge",
    ),
    "mitra-regressor": FoundationModelConfig(
        problem_type="regression",
        ag_model_key="MITRA",
        model_source_uri="autogluon/mitra-regressor",
        model_source_hyperparameter="hf_reg_model",
        inference_hyperparameters={"fine_tune": False},
        predict_instance_type="ml.m5.4xlarge",
        deploy_instance_type="ml.m5.4xlarge",
    ),
    # TabICL, TabDPT-Turbo and Nori download weights pinned by AutoGluon / the model library at fit time, so they
    # have no weights-source hyperparameter and don't support `cache_model_artifact`.
    "tabicl-classifier-v2": FoundationModelConfig(
        problem_type="multiclass",
        ag_model_key="TABICL",
        model_source_uri="jingang/TabICL",
        inference_hyperparameters={"checkpoint_version": "tabicl-classifier-v2-20260212.ckpt"},
        predict_instance_type="ml.g5.xlarge",
    ),
    "tabicl-regressor-v2": FoundationModelConfig(
        problem_type="regression",
        ag_model_key="TABICL",
        model_source_uri="jingang/TabICL",
        inference_hyperparameters={"checkpoint_version": "tabicl-regressor-v2-20260212.ckpt"},
        predict_instance_type="ml.g5.xlarge",
    ),
    "tabdpt-turbo-classifier": FoundationModelConfig(
        problem_type="multiclass",
        ag_model_key="TABDPT-TURBO",
        model_source_uri="Layer6/TabDPT",
        predict_instance_type="ml.g5.xlarge",
    ),
    "tabdpt-turbo-regressor": FoundationModelConfig(
        problem_type="regression",
        ag_model_key="TABDPT-TURBO",
        model_source_uri="Layer6/TabDPT",
        predict_instance_type="ml.g5.xlarge",
    ),
    "nori": FoundationModelConfig(
        problem_type="regression",
        ag_model_key="NORI",
        model_source_uri="Synthefy/Nori",
        inference_hyperparameters={"model": "nori"},
        predict_instance_type="ml.g5.xlarge",
    ),
    "nori-30m": FoundationModelConfig(
        problem_type="regression",
        ag_model_key="NORI",
        model_source_uri="Synthefy/Nori-30M",
        inference_hyperparameters={"model": "nori-30m"},
        predict_instance_type="ml.g5.xlarge",
    ),
}


def get_model_config(model_id: str) -> FoundationModelConfig:
    if model_id not in FOUNDATION_MODEL_REGISTRY:
        available = list(FOUNDATION_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model_id '{model_id}'. Available models: {available}")
    return FOUNDATION_MODEL_REGISTRY[model_id]
