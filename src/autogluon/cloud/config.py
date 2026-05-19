"""Persistent config for AutoGluon-Cloud.

Stores resource identifiers (region, stack name, bucket, IAM role ARN) at
``~/.autogluon/cloud.yaml`` so users don't need to re-specify them every
session. The file contains only non-secret identifiers — no AWS credentials
are ever written to disk.

The file is keyed by backend name so a user can have entries for multiple backends like
``sagemaker`` and ``ray_aws`` configured at the same time::

    sagemaker:
      region: us-east-1
      role_arn: arn:aws:iam::...:role/ag-cloud-sagemaker-execution-role
      bucket: ag-cloud-sagemaker-bucket-...
      stack_name: ag-cloud-sagemaker
    ray_aws:
      region: us-east-1
      role_arn: arn:aws:iam::...:role/ag-cloud-ray-aws-execution-role
      bucket: ag-cloud-ray-aws-bucket-...
      stack_name: ag-cloud-ray-aws
"""

from __future__ import annotations

import os
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml

CONFIG_DIR_ENV = "AG_CONFIG_DIR"


def get_config_dir() -> Path:
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".autogluon"


def get_config_path() -> Path:
    return get_config_dir() / "cloud.yaml"


@dataclass
class BackendConfig:
    """Persisted identifiers for a single AG-Cloud backend."""

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
