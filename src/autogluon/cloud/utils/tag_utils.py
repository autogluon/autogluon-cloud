"""Tag helpers for SageMaker resources created by autogluon-cloud."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

DISABLE_DEFAULT_TAGS_ENV = "AG_CLOUD_DISABLE_DEFAULT_TAGS"


def build_tags(
    module: str,
    extra_tags: Optional[Dict[str, str]] = None,
    user_tags: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Final tags for a SageMaker resource: defaults + extras + user, with user winning on key collision.

    Defaults are skipped entirely when ``AG_CLOUD_DISABLE_DEFAULT_TAGS`` is truthy, so customers in
    tag-restricted AWS orgs can opt out without losing other functionality.
    """
    if os.environ.get(DISABLE_DEFAULT_TAGS_ENV, "").lower() in ("1", "true", "yes"):
        return dict(user_tags or {})
    return {"autogluon-cloud-module": module, **(extra_tags or {}), **(user_tags or {})}


def to_request_tags(tags: Dict[str, str]) -> List[Dict[str, str]]:
    """Convert ``{key: value}`` tags to the list format of SageMaker API requests."""
    return [{"key": key, "value": value} for key, value in tags.items()]
