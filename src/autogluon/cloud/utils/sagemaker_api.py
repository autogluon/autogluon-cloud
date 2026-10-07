"""Helpers for building SageMaker API requests and sending them through a session's boto3 clients."""

import copy
import functools
import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypedDict

from .aws_utils import AwsSession

logger = logging.getLogger(__name__)

# Requests that each method sends, i.e. the valid `backend_overrides` keys, named after the SageMaker API actions.
# `ProductionVariant` is the single variant inside `CreateEndpointConfig`'s `ProductionVariants`.
FIT_OVERRIDE_KEYS = ("CreateTrainingJob",)
DEPLOY_OVERRIDE_KEYS = ("CreateModel", "ProductionVariant", "CreateEndpointConfig", "CreateEndpoint")
BATCH_PREDICT_OVERRIDE_KEYS = ("CreateModel", "CreateTransformJob")
# Fields that link the resources AutoGluon-Cloud creates to each other. Overriding them would point a request at a
# resource we didn't create, which cleanup would then delete.
_RESERVED_OVERRIDE_FIELDS = {
    "ProductionVariant": ("ModelName",),
    "CreateEndpointConfig": ("ProductionVariants",),
    "CreateEndpoint": ("EndpointConfigName",),
    "CreateTransformJob": ("ModelName",),
}

_REMOVED_KWARGS = {
    "backend_kwargs": "`backend_overrides` (and `predictions_path` to choose where `predict()` writes results)",
    "autogluon_sagemaker_estimator_kwargs": "`backend_overrides={'CreateTrainingJob': ...}`",
    "fit_kwargs": "`backend_overrides={'CreateTrainingJob': ...}`",
    "model_kwargs": "`backend_overrides={'CreateModel': ...}`",
    "deploy_kwargs": "`backend_overrides={'ProductionVariant': ..., 'CreateEndpointConfig': ...}`",
    "transformer_kwargs": "`backend_overrides={'CreateTransformJob': ...}`",
    "transform_kwargs": "`backend_overrides={'CreateTransformJob': ...}`",
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


def check_override_keys(overrides: Mapping[str, Any] | None, allowed_keys: Iterable[str]) -> dict[str, Any]:
    """Return ``overrides`` (or ``{}``), raising if it targets a request the calling method doesn't send."""
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(allowed_keys))
    if unknown:
        raise ValueError(f"Unsupported `backend_overrides` key(s) {unknown}. Valid keys: {list(allowed_keys)}.")
    for key, fields in _RESERVED_OVERRIDE_FIELDS.items():
        reserved = sorted(set(overrides.get(key, {})) & set(fields))
        if reserved:
            raise ValueError(f"`backend_overrides[{key!r}]` cannot set {reserved}; AutoGluon-Cloud manages these.")
    return overrides


class TrainingJobKwargs(TypedDict, total=False):
    """Less common settings of methods that run a SageMaker training job, passed as ``**kwargs``."""

    job_name: str
    volume_size: int
    custom_image_uri: str
    timeout: int


class BatchTransformKwargs(TypedDict, total=False):
    """Less common settings of methods that run a SageMaker batch transform job, passed as ``**kwargs``."""

    job_name: str
    instance_count: int
    custom_image_uri: str


class DeployKwargs(TypedDict, total=False):
    """Less common settings of methods that deploy a SageMaker endpoint, passed as ``**kwargs``."""

    initial_instance_count: int
    volume_size: int
    custom_image_uri: str


# Training kwargs that are no longer supported, but accepted with a warning: name -> why the value is ignored.
IGNORED_TRAINING_KWARGS = {
    "instance_count": "only single-instance training is supported",
    "leaderboard": "the leaderboard is always saved",
}
# FoundationModel jobs never save a leaderboard, and never accepted `leaderboard`, so only `instance_count` applies.
IGNORED_FM_JOB_KWARGS = {"instance_count": IGNORED_TRAINING_KWARGS["instance_count"]}


def check_backend_kwargs(
    kwargs: Mapping[str, Any],
    kwargs_type: type,
    method: str,
    ignored: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return ``kwargs`` as a dict, raising ``TypeError`` for keys that the ``kwargs_type`` TypedDict doesn't
    define. Keys in ``ignored`` (name -> reason) are dropped with a warning instead.

    ``kwargs_type`` is annotated as ``type`` because typing has no way to spell "a TypedDict class".
    """
    kwargs = dict(kwargs)
    for name, reason in (ignored or {}).items():
        if name in kwargs:
            kwargs.pop(name)
            logger.warning(f"`{name}` is no longer supported by {method}() and is ignored: {reason}.")
    allowed = kwargs_type.__required_keys__ | kwargs_type.__optional_keys__
    unknown = sorted(set(kwargs) - set(allowed))
    if unknown:
        raise TypeError(
            f"{method}() got unexpected keyword argument(s) {unknown}. Supported keyword arguments: {sorted(allowed)}."
        )
    return kwargs


def delete_quietly(delete: Callable[..., Any], **kwargs) -> None:
    """Call a ``delete_*`` API during rollback, logging instead of raising so the original error propagates."""
    try:
        delete(**kwargs)
    except Exception as e:
        logger.warning(f"Failed to clean up {kwargs}: {e}")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
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
    content_type: str | None = None,
    accept: str | None = None,
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
