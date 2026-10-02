"""Backend settings and persistent resource identifiers for AutoGluon-Cloud.

Stores resource identifiers (region, stack name, bucket, IAM role ARN) at
``~/.autogluon/cloud.yaml`` so users don't need to re-specify them every
session. The file contains only non-secret identifiers — no AWS credentials
are ever written to disk.

The file is keyed by backend name::

    sagemaker:
      region: us-east-1
      role_arn: arn:aws:iam::...:role/ag-cloud-sagemaker-execution-role
      bucket: ag-cloud-sagemaker-bucket-...
      stack_name: ag-cloud-sagemaker
"""

from __future__ import annotations

import os
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

import yaml

CONFIG_DIR_ENV = "AG_CONFIG_DIR"


@dataclass(kw_only=True)
class SageMakerConfig:
    """Reusable SageMaker settings for predictors and foundation models.

    Pass this as ``backend=`` to a cloud predictor or foundation model. Each
    object creates its own backend and jobs; sharing this config does not share
    execution state. Resource sizes and other operation settings remain named
    arguments to ``fit()``, ``predict()`` and ``deploy()``.

    Parameters
    ----------
    region
        AWS region. If omitted, use the region in ``~/.autogluon/cloud.yaml``,
        then the boto3 default region.
    role_arn
        SageMaker execution role ARN. If omitted, use the saved role, then the
        role of the current AWS identity.
    vpc_config
        Networking for training jobs and models, as
        ``{"subnets": [...], "security_group_ids": [...]}``.
    output_kms_key
        KMS key for training artifacts, batch transform outputs, and repacked
        or cached model artifacts in S3.
    volume_kms_key
        KMS key for training, batch transform and realtime endpoint storage
        volumes. Leave unset for instance types with local NVMe storage.
    tags
        Tags added to every SageMaker resource created by this backend.
    """

    name: ClassVar[str] = "sagemaker"
    region: Optional[str] = None
    role_arn: Optional[str] = None
    vpc_config: Optional[Dict[str, List[str]]] = None
    output_kms_key: Optional[str] = None
    volume_kms_key: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


def get_config_dir() -> Path:
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".autogluon"


def get_config_path() -> Path:
    return get_config_dir() / "cloud.yaml"


@dataclass
class BackendConfig:
    """Persisted identifiers for a single AutoGluon-Cloud backend."""

    region: str
    role_arn: str
    bucket: str
    stack_name: Optional[str] = None


@dataclass
class CloudConfig:
    """Top-level config: maps backend name → BackendConfig."""

    backends: Dict[str, BackendConfig] = field(default_factory=dict)


def load_config() -> Optional[CloudConfig]:
    """Load the config file, or return None if it doesn't exist or is empty."""
    path = get_config_path()
    if not path.exists():
        return None
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    if not raw:
        return None
    backends = {name: BackendConfig(**data) for name, data in raw.items()}
    return CloudConfig(backends=backends)


def save_config(config: CloudConfig) -> Path:
    """Persist config atomically with 0600 file perms."""
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: asdict(b) for name, b in config.backends.items()}
    tmp = path.with_suffix(".yaml.tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
    os.replace(tmp, path)
    return path


def delete_config() -> bool:
    """Remove the config file. Returns True if a file was deleted."""
    path = get_config_path()
    if not path.exists():
        return False
    path.unlink()
    return True
