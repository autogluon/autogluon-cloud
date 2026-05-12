"""Foundation model registry.

Maps model_id to AG-compatible configuration for deploy / predict.
"""

from typing import Any, Dict, Literal, TypedDict


class FoundationModelConfig(TypedDict):
    task: Literal["timeseries", "tabular"]
    model_name: str  # AG model class name (e.g. "Chronos", "Chronos2")
    model_config: Dict[str, Any]  # passed to the AG model (e.g. {"model_path": "..."})
    default_instance_type: str
    default_inference_config: Dict[str, Any]  # default prediction kwargs


FOUNDATION_MODEL_REGISTRY: dict[str, FoundationModelConfig] = {
    "chronos-bolt-tiny": {
        "task": "timeseries",
        "model_name": "Chronos",
        "model_config": {"model_path": "amazon/chronos-bolt-tiny"},
        "default_instance_type": "ml.g5.xlarge",
        "default_inference_config": {
            "prediction_length": 64,
            "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
    },
    "chronos-bolt-small": {
        "task": "timeseries",
        "model_name": "Chronos",
        "model_config": {"model_path": "amazon/chronos-bolt-small"},
        "default_instance_type": "ml.g5.xlarge",
        "default_inference_config": {
            "prediction_length": 64,
            "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
    },
    "chronos-bolt-base": {
        "task": "timeseries",
        "model_name": "Chronos",
        "model_config": {"model_path": "amazon/chronos-bolt-base"},
        "default_instance_type": "ml.g5.xlarge",
        "default_inference_config": {
            "prediction_length": 64,
            "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
    },
    "chronos-2": {
        "task": "timeseries",
        "model_name": "Chronos2",
        "model_config": {"model_path": "amazon/chronos-2"},
        "default_instance_type": "ml.g5.xlarge",
        "default_inference_config": {
            "prediction_length": 64,
            "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
    },
}


def get_model_config(model_id: str) -> FoundationModelConfig:
    if model_id not in FOUNDATION_MODEL_REGISTRY:
        available = list(FOUNDATION_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model_id '{model_id}'. Available models: {available}")
    return FOUNDATION_MODEL_REGISTRY[model_id]
