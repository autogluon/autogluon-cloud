"""Tag helpers for SageMaker resources created by autogluon-cloud."""

from __future__ import annotations

import os

DISABLE_DEFAULT_TAGS_ENV = "AG_CLOUD_DISABLE_DEFAULT_TAGS"


def build_tags(
    module: str,
    extra_tags: list[dict[str, str]] | None = None,
    user_tags: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Final tag list for a SageMaker resource: defaults + extras + user, with user winning on key collision.

    Defaults are skipped entirely when ``AG_CLOUD_DISABLE_DEFAULT_TAGS`` is truthy, so customers in
    tag-restricted AWS orgs can opt out without losing other functionality.
    """
    if os.environ.get(DISABLE_DEFAULT_TAGS_ENV, "").lower() in ("1", "true", "yes"):
        return list(user_tags or [])
    base = [{"Key": "autogluon-cloud-module", "Value": module}] + list(extra_tags or [])
    if not user_tags:
        return base
    user_keys = {t["Key"] for t in user_tags}
    return [t for t in base if t["Key"] not in user_keys] + list(user_tags)
