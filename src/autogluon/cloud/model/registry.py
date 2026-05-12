"""Foundation model registry.

Maps model_id to AG-compatible configuration for deploy / predict.
"""

from typing import Any, Dict, Literal, TypedDict


class FoundationModelConfig(TypedDict):
    task: Literal["forecasting", "classification", "regression"]
    model_name: str  # AG model class name (e.g. "Chronos", "Chronos2", "Mitra")
    model_config: Dict[str, Any]  # passed to the AG model (e.g. {"model_path": "..."})
    default_instance_type: str


FOUNDATION_MODEL_REGISTRY: dict[str, FoundationModelConfig] = {
    "chronos-bolt-tiny": {
        "task": "forecasting",
        "model_name": "Chronos",
        "model_config": {"model_path": "amazon/chronos-bolt-tiny"},
        "default_instance_type": "ml.g5.xlarge",
    },
    "chronos-bolt-small": {
        "task": "forecasting",
        "model_name": "Chronos",
        "model_config": {"model_path": "amazon/chronos-bolt-small"},
        "default_instance_type": "ml.g5.xlarge",
    },
    "chronos-bolt-base": {
        "task": "forecasting",
        "model_name": "Chronos",
        "model_config": {"model_path": "amazon/chronos-bolt-base"},
        "default_instance_type": "ml.g5.xlarge",
    },
    "chronos-2": {
        "task": "forecasting",
        "model_name": "Chronos2",
        "model_config": {"model_path": "amazon/chronos-2"},
        "default_instance_type": "ml.g5.xlarge",
    },
    "mitra-classification": {
        "task": "classification",
        "model_name": "Mitra",
        "model_config": {"model_path": "TODO"},
        "default_instance_type": "ml.m5.xlarge",
    },
    "mitra-regression": {
        "task": "regression",
        "model_name": "Mitra",
        "model_config": {"model_path": "TODO"},
        "default_instance_type": "ml.m5.xlarge",
    },
}


def get_model_config(model_id: str) -> FoundationModelConfig:
    if model_id not in FOUNDATION_MODEL_REGISTRY:
        available = list(FOUNDATION_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model_id '{model_id}'. Available models: {available}")
    return FOUNDATION_MODEL_REGISTRY[model_id]
