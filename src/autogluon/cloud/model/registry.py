"""Foundation model registry.

Maps model_id to AG-compatible configuration for deploy / predict.
"""

from typing import Any, Dict, Literal, TypedDict


class FoundationModelConfig(TypedDict):
    task: Literal["forecasting", "classification", "regression"]
    model_name: str  # AG model class name (e.g. "Chronos", "Chronos2", "Mitra")
    inference_hyperparameters: Dict[str, Any]  # defaults for deploy() and predict()
    training_hyperparameters: Dict[str, Any]  # defaults for fit()
    predict_instance_type: str  # batch predict
    deploy_instance_type: str  # real-time endpoint
    fit_instance_type: str  # fine-tuning
    fine_tunable: bool  # whether .fit() is supported


FOUNDATION_MODEL_REGISTRY: dict[str, FoundationModelConfig] = {
    "chronos-bolt-tiny": {
        "task": "forecasting",
        "model_name": "Chronos",
        "inference_hyperparameters": {"model_path": "amazon/chronos-bolt-tiny"},
        "training_hyperparameters": {"model_path": "amazon/chronos-bolt-tiny"},
        "predict_instance_type": "ml.m5.2xlarge",
        "deploy_instance_type": "ml.g5.xlarge",
        "fit_instance_type": "ml.g5.xlarge",
        "fine_tunable": False,
    },
    "chronos-bolt-small": {
        "task": "forecasting",
        "model_name": "Chronos",
        "inference_hyperparameters": {"model_path": "amazon/chronos-bolt-small"},
        "training_hyperparameters": {"model_path": "amazon/chronos-bolt-small"},
        "predict_instance_type": "ml.m5.2xlarge",
        "deploy_instance_type": "ml.g5.xlarge",
        "fit_instance_type": "ml.g5.xlarge",
        "fine_tunable": False,
    },
    "chronos-bolt-base": {
        "task": "forecasting",
        "model_name": "Chronos",
        "inference_hyperparameters": {"model_path": "amazon/chronos-bolt-base"},
        "training_hyperparameters": {"model_path": "amazon/chronos-bolt-base"},
        "predict_instance_type": "ml.m5.2xlarge",
        "deploy_instance_type": "ml.g5.xlarge",
        "fit_instance_type": "ml.g5.xlarge",
        "fine_tunable": False,
    },
    "chronos-2": {
        "task": "forecasting",
        "model_name": "Chronos2",
        "inference_hyperparameters": {"model_path": "amazon/chronos-2"},
        "training_hyperparameters": {"model_path": "amazon/chronos-2", "fine_tune": True},
        "predict_instance_type": "ml.m5.2xlarge",
        "deploy_instance_type": "ml.g5.xlarge",
        "fit_instance_type": "ml.g5.xlarge",
        "fine_tunable": True,
    },
    "mitra-classification": {
        "task": "classification",
        "model_name": "MITRA",
        "inference_hyperparameters": {"fine_tune": False},
        "training_hyperparameters": {"fine_tune": True},
        "predict_instance_type": "ml.g5.xlarge",
        "deploy_instance_type": "ml.g5.xlarge",
        "fit_instance_type": "ml.g5.xlarge",
        "fine_tunable": True,
    },
    "mitra-regression": {
        "task": "regression",
        "model_name": "MITRA",
        "inference_hyperparameters": {"fine_tune": False},
        "training_hyperparameters": {"fine_tune": True},
        "predict_instance_type": "ml.g5.xlarge",
        "deploy_instance_type": "ml.g5.xlarge",
        "fit_instance_type": "ml.g5.xlarge",
        "fine_tunable": True,
    },
    "tabicl": {
        "task": "classification",
        "model_name": "TABICL",
        "inference_hyperparameters": {},
        "training_hyperparameters": {},
        "predict_instance_type": "ml.g5.xlarge",
        "deploy_instance_type": "ml.g5.xlarge",
        "fit_instance_type": "ml.g5.xlarge",
        "fine_tunable": False,
    },
}


def get_model_config(model_id: str) -> FoundationModelConfig:
    if model_id not in FOUNDATION_MODEL_REGISTRY:
        available = list(FOUNDATION_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model_id '{model_id}'. Available models: {available}")
    return FOUNDATION_MODEL_REGISTRY[model_id]
