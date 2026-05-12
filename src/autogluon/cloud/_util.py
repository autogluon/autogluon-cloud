"""Shared helpers used by both the Python setup API and the CLI commands."""

from __future__ import annotations

from importlib import resources
from importlib.abc import Traversable
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from rich.console import Console

console = Console()


SUPPORTED_BACKENDS = ("sagemaker", "ray_aws")


def get_template_path(backend: str) -> Traversable:
    """Return a handle to the bundled CloudFormation template."""
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Supported: {SUPPORTED_BACKENDS}")
    filename = f"ag_cloud_{backend}.yaml"
    return resources.files("autogluon.cloud.templates").joinpath(filename)


def make_boto_session(
    aws_profile: Optional[str] = None,
    region: Optional[str] = None,
) -> boto3.Session:
    """Build a boto3 Session honoring an optional named AWS profile."""
    kwargs = {}
    if aws_profile:
        kwargs["profile_name"] = aws_profile
    if region:
        kwargs["region_name"] = region
    return boto3.Session(**kwargs)


def detect_aws_identity(
    region: Optional[str] = None,
    aws_profile: Optional[str] = None,
) -> Optional[dict]:
    """Return the caller's AWS identity via STS, or None if creds are missing."""
    try:
        session = make_boto_session(aws_profile=aws_profile, region=region)
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        return {
            "account": identity["Account"],
            "arn": identity["Arn"],
            "region": session.region_name,
        }
    except (NoCredentialsError, ClientError, BotoCoreError):
        return None


def format_identity(identity: dict) -> str:
    return f"account [bold]{identity['account']}[/bold], region [bold]{identity['region'] or 'unset'}[/bold]"
