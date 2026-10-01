"""Helpers for calling SageMaker through sagemaker-core (SageMaker Python SDK v3) resource classes.

AutoGluon-Cloud builds each SageMaker API request as a plain snake_case dict (the sagemaker-core field
names, i.e. the AWS API fields in snake_case), merges the user's ``sagemaker_overrides`` on top, and
passes the result to the corresponding ``sagemaker.core.resources.<Resource>.create()`` call, which
validates it against the typed shapes before sending.
"""

from __future__ import annotations

import copy
import functools
import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import boto3
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Keys accepted in ``sagemaker_overrides`` for each public method. Every key names the request it is merged into.
# ``production_variant`` is the single ``ProductionVariant`` inside ``create_endpoint_config.production_variants``;
# it gets its own key because deep-merging into a list is ambiguous.
FIT_OVERRIDE_KEYS = ("create_training_job",)
DEPLOY_OVERRIDE_KEYS = ("create_model", "production_variant", "create_endpoint_config", "create_endpoint")
BATCH_PREDICT_OVERRIDE_KEYS = ("create_model", "create_transform_job")

# Removed SageMaker SDK v2 passthrough kwargs and where their contents moved.
_LEGACY_KWARG_HINTS = {
    "backend_kwargs": (
        "`backend_kwargs` was removed when AutoGluon-Cloud migrated to SageMaker SDK v3. Use the named arguments "
        "(e.g. `environment`, `use_spot_instances`, `download`, `save_path`), the predictor constructor "
        "(`vpc_config`, `kms_key`, `tags`), or `sagemaker_overrides` for raw SageMaker API request fields."
    ),
    "custom_image_uri": "`custom_image_uri` was renamed to `image_uri`.",
    "autogluon_sagemaker_estimator_kwargs": (
        "SageMaker SDK v3 has no Estimator. Use `sagemaker_overrides={'create_training_job': {...}}`."
    ),
    "fit_kwargs": "SageMaker SDK v3 has no Estimator. Use `sagemaker_overrides={'create_training_job': {...}}`.",
    "model_kwargs": "Use `sagemaker_overrides={'create_model': {...}}` and `environment=`.",
    "deploy_kwargs": (
        "Use `sagemaker_overrides={'production_variant': {...}, 'create_endpoint_config': {...}, "
        "'create_endpoint': {...}}`."
    ),
    "transformer_kwargs": "Use `sagemaker_overrides={'create_transform_job': {...}}`.",
    "transform_kwargs": "Use `sagemaker_overrides={'create_transform_job': {...}}`.",
}


def reject_legacy_kwargs(func):
    """Turn removed v2-era kwargs into an actionable ``TypeError`` instead of Python's generic one."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        legacy = [name for name in kwargs if name in _LEGACY_KWARG_HINTS]
        if legacy:
            hints = " ".join(_LEGACY_KWARG_HINTS[name] for name in legacy)
            raise TypeError(f"{func.__qualname__}() got removed keyword argument(s) {legacy}. {hints}")
        return func(*args, **kwargs)

    return wrapper


def _to_plain(value: Any) -> Any:
    """Recursively convert sagemaker-core shape objects to plain snake_case dicts (only explicitly set fields)."""
    if isinstance(value, BaseModel):
        return {k: _to_plain(v) for k, v in value.model_dump(exclude_unset=True).items()}
    if isinstance(value, Mapping):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge ``override`` into a copy of ``base``: dicts merge recursively, every other value replaces."""
    merged = copy.deepcopy(dict(base))
    for key, value in _to_plain(override).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_sagemaker_overrides(
    overrides: Optional[Mapping[str, Any]], allowed_keys: Iterable[str]
) -> Dict[str, Dict[str, Any]]:
    """Check that ``overrides`` only targets requests the calling method actually sends."""
    if not overrides:
        return {}
    allowed_keys = tuple(allowed_keys)
    unknown = sorted(set(overrides) - set(allowed_keys))
    if unknown:
        raise ValueError(f"Unsupported `sagemaker_overrides` key(s) {unknown}. Valid keys here: {list(allowed_keys)}.")
    for key, value in overrides.items():
        if not isinstance(value, (Mapping, BaseModel)):
            raise TypeError(f"`sagemaker_overrides[{key!r}]` must be a dict, got {type(value).__name__}.")
    return {key: _to_plain(value) for key, value in overrides.items()}


def apply_overrides(request: Dict[str, Any], overrides: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """Return ``request`` with ``overrides[key]`` (if any) deep-merged on top."""
    if key not in overrides:
        return request
    logger.debug(f"Applying sagemaker_overrides[{key!r}]: {overrides[key]}")
    return deep_merge(request, overrides[key])


def normalize_tags(tags: Optional[Union[Mapping[str, str], List[Dict[str, str]]]]) -> List[Dict[str, str]]:
    """Accept user tags as ``{"key": "value"}`` (or a ``[{"Key", "Value"}]`` list) and return the list form."""
    if not tags:
        return []
    if isinstance(tags, Mapping):
        return [{"Key": str(k), "Value": str(v)} for k, v in tags.items()]
    return [{"Key": t["Key"], "Value": t["Value"]} for t in tags]


def to_request_tags(tags: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert ``[{"Key", "Value"}]`` tags to the sagemaker-core ``Tag`` field names."""
    return [{"key": t["Key"], "value": t["Value"]} for t in tags]


def normalize_vpc_config(vpc_config: Optional[Mapping[str, Any]]) -> Optional[Dict[str, List[str]]]:
    """Validate a user ``vpc_config`` of the form ``{"subnets": [...], "security_group_ids": [...]}``."""
    if vpc_config is None:
        return None
    expected = {"subnets", "security_group_ids"}
    if set(vpc_config) != expected:
        raise ValueError(
            f"`vpc_config` must have exactly the keys {sorted(expected)}, got {sorted(vpc_config)}. "
            "Example: {'subnets': ['subnet-123'], 'security_group_ids': ['sg-123']}."
        )
    return {key: list(vpc_config[key]) for key in sorted(expected)}


def bind_core_session(boto_session: boto3.Session) -> None:
    """Point sagemaker-core at ``boto_session``.

    sagemaker-core caches one set of boto clients per process (``SageMakerClient`` is a singleton) and ignores the
    ``session`` argument of resource methods once that cache exists. Several AutoGluon-Cloud objects can use
    different sessions (e.g. an endpoint handle created with an explicit ``session``), so we rebuild the cached
    clients whenever the requested session differs from the cached one. Not thread-safe across sessions.
    """
    from sagemaker.core.utils.utils import SageMakerClient, SingletonMeta

    current = SingletonMeta._instances.get(SageMakerClient)
    if current is not None and current.session is boto_session and current.region_name == boto_session.region_name:
        return
    SingletonMeta._instances.pop(SageMakerClient, None)
    SageMakerClient(session=boto_session, region_name=boto_session.region_name)


def script_mode_environment(entry_point: str, region: str) -> Dict[str, str]:
    """Environment variables telling the AutoGluon DLC's inference toolkit which bundled script to load.

    The model tarball always carries the serving code under ``code/`` (see ``SagemakerBackend``), which SageMaker
    extracts to ``/opt/ml/model/code``.
    """
    return {
        "SAGEMAKER_PROGRAM": os.path.basename(entry_point),
        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
        "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
        "SAGEMAKER_REGION": region,
    }


def invoke_endpoint(
    endpoint_name: str,
    boto_session: boto3.Session,
    payload: Any,
    serializer,
    deserializer,
    content_type: Optional[str] = None,
    accept: Optional[str] = None,
) -> Any:
    """Serialize ``payload``, invoke a SageMaker endpoint, and deserialize the response.

    ``content_type`` / ``accept`` default to the serializer's ``CONTENT_TYPE`` and the deserializer's ``ACCEPT``.
    """
    from sagemaker.core.resources import Endpoint

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
    """Delete a SageMaker endpoint together with its endpoint config and the models it serves."""
    from sagemaker.core.resources import Endpoint, EndpointConfig, Model

    bind_core_session(boto_session)
    endpoint = Endpoint.get(endpoint_name=endpoint_name, session=boto_session, region=boto_session.region_name)
    endpoint_config = EndpointConfig.get(
        endpoint_config_name=endpoint.endpoint_config_name, session=boto_session, region=boto_session.region_name
    )
    model_names = [variant.model_name for variant in endpoint_config.production_variants]

    logger.info(f"Deleting endpoint {endpoint_name}")
    endpoint.delete()
    endpoint_config.delete()
    for model_name in model_names:
        logger.info(f"Deleting endpoint model {model_name}")
        Model(model_name=model_name).delete()
