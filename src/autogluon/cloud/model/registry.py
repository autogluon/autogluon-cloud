"""Foundation model registry.

Maps model_id to AG-compatible configuration for deploy / predict.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal


@dataclass(frozen=True)
class FoundationModelConfig:
    task: Literal["forecasting", "classification", "regression"]
    ag_model_key: str  # key in the AG hyperparameters dict (e.g. "Chronos", "Chronos2", "Mitra")
    model_source_uri: str  # where weights are downloaded from (e.g. "autogluon/chronos-2")
    predict_instance_type: str = "ml.m5.2xlarge"  # batch predict
    deploy_instance_type: str = "ml.g5.xlarge"  # real-time endpoint
    fit_instance_type: str = "ml.g5.xlarge"  # fine-tuning
    inference_hyperparameters: Dict[str, Any] = field(default_factory=dict)  # defaults for deploy() and predict()
    training_hyperparameters: Dict[str, Any] = field(default_factory=dict)  # defaults for fit()
    fine_tunable: bool = False  # whether .fit() is supported


FOUNDATION_MODEL_REGISTRY: Dict[str, FoundationModelConfig] = {
    "chronos-bolt-tiny": FoundationModelConfig(
        task="forecasting",
        ag_model_key="Chronos",
        model_source_uri="autogluon/chronos-bolt-tiny",
    ),
    "chronos-bolt-small": FoundationModelConfig(
        task="forecasting",
        ag_model_key="Chronos",
        model_source_uri="autogluon/chronos-bolt-small",
    ),
    "chronos-bolt-base": FoundationModelConfig(
        task="forecasting",
        ag_model_key="Chronos",
        model_source_uri="autogluon/chronos-bolt-base",
    ),
    "chronos-2-small": FoundationModelConfig(
        task="forecasting",
        ag_model_key="Chronos2",
        model_source_uri="autogluon/chronos-2-small",
    ),
    "chronos-2": FoundationModelConfig(
        task="forecasting",
        ag_model_key="Chronos2",
        model_source_uri="autogluon/chronos-2",
    ),
}


def get_model_config(model_id: str) -> FoundationModelConfig:
    if model_id not in FOUNDATION_MODEL_REGISTRY:
        available = list(FOUNDATION_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model_id '{model_id}'. Available models: {available}")
    return FOUNDATION_MODEL_REGISTRY[model_id]
