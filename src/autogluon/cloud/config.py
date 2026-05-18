"""Persistent config for AutoGluon-Cloud.

Stores resource identifiers (region, stack name, bucket, IAM role ARN) at
``~/.autogluon/cloud.yaml`` so users don't need to re-specify them every
session. The file contains only non-secret identifiers — no AWS credentials
are ever written to disk.
"""

from __future__ import annotations

import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import yaml

CONFIG_DIR_ENV = "AUTOGLUON_CLOUD_CONFIG_DIR"


def get_config_dir() -> Path:
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".autogluon"


def get_config_path() -> Path:
    return get_config_dir() / "cloud.yaml"


@dataclass
class CloudConfig:
    region: str
    role_arn: str
    bucket: str
    backend: str = "sagemaker"
    stack_name: Optional[str] = None


def load_config() -> Optional[CloudConfig]:
    """Load the config file, or return None if it doesn't exist."""
    path = get_config_path()
    if not path.exists():
        return None
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    if not raw:
        return None
    return CloudConfig(**raw)


def save_config(config: CloudConfig) -> Path:
    """Persist config atomically with 0600 file perms."""
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)
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
