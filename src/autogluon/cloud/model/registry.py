"""Foundation model registry.

Maps model_id to AG-compatible configuration for deploy / predict.
"""

from typing import Any, Dict, Literal, TypedDict


class FoundationModelConfig(TypedDict):
    task: Literal["forecasting", "classification", "regression"]
    model_name: str  # AG model class name (e.g. "Chronos", "Chronos2", "Mitra")
    model_source_uri: str  # where weights are downloaded from (e.g. "amazon/chronos-2")
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
        "model_source_uri": "amazon/chronos-bolt-tiny",
        "inference_hyperparameters": {},
        "training_hyperparameters": {},
        "predict_instance_type": "ml.m5.2xlarge",
        "deploy_instance_type": "ml.g5.xlarge",
        "fit_instance_type": "ml.g5.xlarge",
        "fine_tunable": False,
    },
    "chronos-bolt-small": {
        "task": "forecasting",
        "model_name": "Chronos",
        "model_source_uri": "amazon/chronos-bolt-small",
        "inference_hyperparameters": {},
        "training_hyperparameters": {},
        "predict_instance_type": "ml.m5.2xlarge",
        "deploy_instance_type": "ml.g5.xlarge",
        "fit_instance_type": "ml.g5.xlarge",
        "fine_tunable": False,
    },
    "chronos-bolt-base": {
        "task": "forecasting",
        "model_name": "Chronos",
        "model_source_uri": "amazon/chronos-bolt-base",
        "inference_hyperparameters": {},
        "training_hyperparameters": {},
        "predict_instance_type": "ml.m5.2xlarge",
        "deploy_instance_type": "ml.g5.xlarge",
        "fit_instance_type": "ml.g5.xlarge",
        "fine_tunable": False,
    },
    "chronos-2": {
        "task": "forecasting",
        "model_name": "Chronos2",
        "model_source_uri": "amazon/chronos-2",
        "inference_hyperparameters": {},
        "training_hyperparameters": {"fine_tune": True},
        "predict_instance_type": "ml.m5.2xlarge",
        "deploy_instance_type": "ml.g5.xlarge",
        "fit_instance_type": "ml.g5.xlarge",
        "fine_tunable": True,
    },
    # TODO: Replace dummy configs with real values
    "mitra-classification": {
        "task": "classification",
        "model_name": "Mitra",
        "model_source_uri": "TODO",
        "inference_hyperparameters": {},
        "training_hyperparameters": {},
        "predict_instance_type": "ml.m5.2xlarge",
        "deploy_instance_type": "ml.g5.xlarge",
        "fit_instance_type": "ml.g5.xlarge",
        "fine_tunable": False,
    },
    "mitra-regression": {
        "task": "regression",
        "model_name": "Mitra",
        "model_source_uri": "TODO",
        "inference_hyperparameters": {},
        "training_hyperparameters": {},
        "predict_instance_type": "ml.m5.2xlarge",
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
