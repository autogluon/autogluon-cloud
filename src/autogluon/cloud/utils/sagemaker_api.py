"""Helpers for building SageMaker API requests and calling them through sagemaker-core."""

import copy
import functools
import logging
from typing import Any, Dict, Iterable, Mapping, Optional

import boto3
from sagemaker.core.resources import Endpoint, EndpointConfig, Model

from .sagemaker_core_workarounds import bind_core_session

logger = logging.getLogger(__name__)

# Requests that each method sends, i.e. the valid `sagemaker_overrides` keys. `production_variant` is the single
# variant inside `create_endpoint_config.production_variants`.
FIT_OVERRIDE_KEYS = ("create_training_job",)
DEPLOY_OVERRIDE_KEYS = ("create_model", "production_variant", "create_endpoint_config", "create_endpoint")
BATCH_PREDICT_OVERRIDE_KEYS = ("create_model", "create_transform_job")

_REMOVED_KWARGS = {
    "backend_kwargs": "named arguments, the constructor's `vpc_config` / `kms_key` / `tags`, or `sagemaker_overrides`",
    "custom_image_uri": "`image_uri`",
    "autogluon_sagemaker_estimator_kwargs": "`sagemaker_overrides={'create_training_job': ...}`",
    "fit_kwargs": "`sagemaker_overrides={'create_training_job': ...}`",
    "model_kwargs": "`environment` or `sagemaker_overrides={'create_model': ...}`",
    "deploy_kwargs": "`sagemaker_overrides={'production_variant': ..., 'create_endpoint_config': ...}`",
    "transformer_kwargs": "`sagemaker_overrides={'create_transform_job': ...}`",
    "transform_kwargs": "`sagemaker_overrides={'create_transform_job': ...}`",
}


def reject_legacy_kwargs(func):
    """Raise an actionable ``TypeError`` for kwargs removed in the SageMaker SDK v3 migration."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for name in kwargs:
            if name in _REMOVED_KWARGS:
                raise TypeError(
                    f"`{name}` was removed from {func.__qualname__}(). Use {_REMOVED_KWARGS[name]} instead."
                )
        return func(*args, **kwargs)

    return wrapper


def check_override_keys(overrides: Optional[Mapping[str, Any]], allowed_keys: Iterable[str]) -> Dict[str, Any]:
    """Return ``overrides`` (or ``{}``), raising if it targets a request the calling method doesn't send."""
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(allowed_keys))
    if unknown:
        raise ValueError(f"Unsupported `sagemaker_overrides` key(s) {unknown}. Valid keys: {list(allowed_keys)}.")
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
    boto_session: boto3.Session,
    payload: Any,
    serializer,
    deserializer,
    content_type: Optional[str] = None,
    accept: Optional[str] = None,
) -> Any:
    """Serialize ``payload``, invoke the endpoint, and deserialize the response."""
    bind_core_session(boto_session)
    response = Endpoint(endpoint_name=endpoint_name).invoke(
        body=serializer.serialize(payload),
        content_type=content_type or serializer.CONTENT_TYPE,
        accept=accept or ", ".join(deserializer.ACCEPT),
        session=boto_session,
        region=boto_session.region_name,
    )
    return deserializer.deserialize(response.body, response.content_type)


def delete_endpoint(endpoint_name: str, boto_session: boto3.Session) -> None:
    """Delete an endpoint together with its endpoint config and models."""
    bind_core_session(boto_session)
    region = boto_session.region_name
    endpoint = Endpoint.get(endpoint_name=endpoint_name, session=boto_session, region=region)
    endpoint_config = EndpointConfig.get(
        endpoint_config_name=endpoint.endpoint_config_name, session=boto_session, region=region
    )
    logger.info(f"Deleting endpoint {endpoint_name}")
    endpoint.delete()
    endpoint_config.delete()
    for variant in endpoint_config.production_variants:
        Model(model_name=variant.model_name).delete()
