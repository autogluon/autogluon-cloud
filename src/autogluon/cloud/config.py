"""Persistent config for AutoGluon-Cloud.

Stores per-profile information (region, stack name, bucket, IAM role ARN) at
``~/.autogluon/cloud.yaml`` so users don't need to re-specify these every
session. The file contains only non-secret identifiers — no AWS credentials
are ever written to disk.
"""

from __future__ import annotations

import logging
import os
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)

CONFIG_VERSION = 1
DEFAULT_PROFILE = "default"
CONFIG_DIR_ENV = "AUTOGLUON_CLOUD_CONFIG_DIR"
PROFILE_ENV = "AUTOGLUON_CLOUD_PROFILE"


def get_config_dir() -> Path:
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".autogluon"


def get_config_path() -> Path:
    return get_config_dir() / "cloud.yaml"


@dataclass
class Profile:
    region: str
    role_arn: str
    bucket: str
    backend: str = "sagemaker"
    stack_name: Optional[str] = None
    aws_profile: Optional[str] = None


@dataclass
class CloudConfig:
    version: int = CONFIG_VERSION
    active_profile: str = DEFAULT_PROFILE
    profiles: Dict[str, Profile] = field(default_factory=dict)

    def get_profile(self, name: Optional[str] = None) -> Profile:
        name = name or os.environ.get(PROFILE_ENV) or self.active_profile
        if name not in self.profiles:
            raise KeyError(
                f"Profile {name!r} not found in {get_config_path()}. Available profiles: {sorted(self.profiles)}"
            )
        return self.profiles[name]


def load_config() -> Optional[CloudConfig]:
    """Load the config file, or return None if it doesn't exist.

    Returns None (rather than raising) so callers can gracefully fall back to
    requiring explicit ``cloud_output_path`` and role ARN.
    """
    path = get_config_path()
    if not path.exists():
        return None
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    profiles = {name: Profile(**data) for name, data in (raw.get("profiles") or {}).items()}
    return CloudConfig(
        version=raw.get("version", CONFIG_VERSION),
        active_profile=raw.get("active_profile", DEFAULT_PROFILE),
        profiles=profiles,
    )


def save_config(config: CloudConfig) -> Path:
    """Persist config atomically with 0600 file perms."""
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": config.version,
        "active_profile": config.active_profile,
        "profiles": {name: asdict(p) for name, p in config.profiles.items()},
    }
    tmp = path.with_suffix(".yaml.tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
    os.replace(tmp, path)
    return path


def upsert_profile(name: str, profile: Profile, set_active: bool = True) -> CloudConfig:
    config = load_config() or CloudConfig()
    config.profiles[name] = profile
    if set_active or len(config.profiles) == 1:
        config.active_profile = name
    save_config(config)
    return config


def delete_profile(name: str) -> bool:
    config = load_config()
    if config is None or name not in config.profiles:
        return False
    del config.profiles[name]
    if config.active_profile == name:
        config.active_profile = next(iter(config.profiles), DEFAULT_PROFILE)
    save_config(config)
    return True
