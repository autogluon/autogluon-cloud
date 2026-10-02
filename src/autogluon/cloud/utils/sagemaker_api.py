"""Helpers for building SageMaker API requests and sending them through a session's boto3 clients."""

import copy
import functools
import logging
from typing import Any, Dict, Iterable, Mapping, Optional

from .aws_utils import AwsSession

logger = logging.getLogger(__name__)

# Requests that each method sends, i.e. the valid `backend_overrides` keys, named after the boto3 client methods.
# `production_variant` is the single variant inside `create_endpoint_config`'s `ProductionVariants`.
FIT_OVERRIDE_KEYS = ("create_training_job",)
DEPLOY_OVERRIDE_KEYS = ("create_model", "production_variant", "create_endpoint_config", "create_endpoint")
BATCH_PREDICT_OVERRIDE_KEYS = ("create_model", "create_transform_job")

_REMOVED_KWARGS = {
    "backend_kwargs": "`backend_overrides` (and `predictions_path` to choose where `predict()` writes results)",
    "autogluon_sagemaker_estimator_kwargs": "`backend_overrides={'create_training_job': ...}`",
    "fit_kwargs": "`backend_overrides={'create_training_job': ...}`",
    "model_kwargs": "`backend_overrides={'create_model': ...}`",
    "deploy_kwargs": "`backend_overrides={'production_variant': ..., 'create_endpoint_config': ...}`",
    "transformer_kwargs": "`backend_overrides={'create_transform_job': ...}`",
    "transform_kwargs": "`backend_overrides={'create_transform_job': ...}`",
}


def reject_legacy_kwargs(func):
    """Raise an actionable ``TypeError`` for kwargs removed when AutoGluon-Cloud stopped using the SageMaker Python SDK."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for name, value in kwargs.items():
            if name not in _REMOVED_KWARGS:
                continue
            if _sets_custom_entry_point(value):
                raise TypeError(
                    f"Custom `entry_point` / `source_dir` scripts (passed via `{name}`) are no longer supported: "
                    "AutoGluon-Cloud always runs its own training and serving scripts. To customize the container, "
                    "pass `custom_image_uri`."
                )
            raise TypeError(f"`{name}` was removed from {func.__qualname__}(). Use {_REMOVED_KWARGS[name]} instead.")
        return func(*args, **kwargs)

    return wrapper


def _sets_custom_entry_point(value: Any) -> bool:
    """Whether a legacy SDK kwargs dict (possibly nested, e.g. ``backend_kwargs["model_kwargs"]``) sets a script."""
    if not isinstance(value, Mapping):
        return False
    return any(key in ("entry_point", "source_dir") or _sets_custom_entry_point(v) for key, v in value.items())


def check_override_keys(overrides: Optional[Mapping[str, Any]], allowed_keys: Iterable[str]) -> Dict[str, Any]:
    """Return ``overrides`` (or ``{}``), raising if it targets a request the calling method doesn't send."""
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(allowed_keys))
    if unknown:
        raise ValueError(f"Unsupported `backend_overrides` key(s) {unknown}. Valid keys: {list(allowed_keys)}.")
    return overrides


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge ``override`` into a copy of ``base``: dicts merge recursively, every other value replaces."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def invoke_endpoint(
    endpoint_name: str,
    session: AwsSession,
    payload: Any,
    serializer,
    deserializer,
    content_type: Optional[str] = None,
    accept: Optional[str] = None,
) -> Any:
    """Serialize ``payload``, invoke the endpoint, and deserialize the response."""
    response = session.sagemaker_runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=serializer.serialize(payload),
        ContentType=content_type or serializer.content_type,
        Accept=accept or ", ".join(deserializer.accept),
    )
    return deserializer.deserialize(response["Body"], response["ContentType"])


def delete_endpoint(endpoint_name: str, session: AwsSession) -> None:
    """Delete an endpoint together with its endpoint config and models."""
    client = session.sagemaker_client
    endpoint = client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_config = client.describe_endpoint_config(EndpointConfigName=endpoint["EndpointConfigName"])
    logger.info(f"Deleting endpoint {endpoint_name}")
    client.delete_endpoint(EndpointName=endpoint_name)
    client.delete_endpoint_config(EndpointConfigName=endpoint["EndpointConfigName"])
    for variant in endpoint_config["ProductionVariants"]:
        client.delete_model(ModelName=variant["ModelName"])
