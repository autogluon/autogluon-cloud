"""Foundation model registry.

Maps model_id to AG-compatible configuration for deploy / predict.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal


@dataclass(frozen=True)
class FoundationModelConfig:
    task: Literal["forecasting", "classification", "regression"]
    ag_model_key: str  # key in the AG hyperparameters dict (e.g. "Chronos", "Chronos2", "Mitra")
    model_source_uri: str  # where weights are downloaded from (e.g. "amazon/chronos-2")
    predict_instance_type: str  # batch predict
    deploy_instance_type: str  # real-time endpoint
    fit_instance_type: str  # fine-tuning
    inference_hyperparameters: Dict[str, Any] = field(default_factory=dict)  # defaults for deploy() and predict()
    training_hyperparameters: Dict[str, Any] = field(default_factory=dict)  # defaults for fit()
    fine_tunable: bool = False  # whether .fit() is supported


_DEFAULT_INSTANCE_TYPES = {
    "predict_instance_type": "ml.m5.2xlarge",
    "deploy_instance_type": "ml.g5.xlarge",
    "fit_instance_type": "ml.g5.xlarge",
}


FOUNDATION_MODEL_REGISTRY: Dict[str, FoundationModelConfig] = {
    "chronos-bolt-tiny": FoundationModelConfig(
        task="forecasting",
        ag_model_key="Chronos",
        model_source_uri="amazon/chronos-bolt-tiny",
        **_DEFAULT_INSTANCE_TYPES,
    ),
    "chronos-bolt-small": FoundationModelConfig(
        task="forecasting",
        ag_model_key="Chronos",
        model_source_uri="amazon/chronos-bolt-small",
        **_DEFAULT_INSTANCE_TYPES,
    ),
    "chronos-bolt-base": FoundationModelConfig(
        task="forecasting",
        ag_model_key="Chronos",
        model_source_uri="amazon/chronos-bolt-base",
        **_DEFAULT_INSTANCE_TYPES,
    ),
    "chronos-2": FoundationModelConfig(
        task="forecasting",
        ag_model_key="Chronos2",
        model_source_uri="amazon/chronos-2",
        training_hyperparameters={"fine_tune": True},
        fine_tunable=True,
        **_DEFAULT_INSTANCE_TYPES,
    ),
    # TODO: Replace dummy configs with real values
    "mitra-classification": FoundationModelConfig(
        task="classification",
        ag_model_key="Mitra",
        model_source_uri="TODO",
        **_DEFAULT_INSTANCE_TYPES,
    ),
    "mitra-regression": FoundationModelConfig(
        task="regression",
        ag_model_key="Mitra",
        model_source_uri="TODO",
        **_DEFAULT_INSTANCE_TYPES,
    ),
}


def get_model_config(model_id: str) -> FoundationModelConfig:
    if model_id not in FOUNDATION_MODEL_REGISTRY:
        available = list(FOUNDATION_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model_id '{model_id}'. Available models: {available}")
    return FOUNDATION_MODEL_REGISTRY[model_id]
