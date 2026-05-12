"""Python API for provisioning AutoGluon-Cloud on AWS.

Usage::

    import autogluon.cloud as agc

    agc.initialize(backend="sagemaker", region="us-east-1")   # deploy CFN + save config
    agc.initialize(role_arn=..., bucket=...)                  # save config, skip CFN
    agc.status()                                             # dict of health checks
    agc.teardown(delete_bucket_contents=True)                # delete CFN + config
"""

from __future__ import annotations

import typing
from importlib import resources
from typing import Any, Dict, Literal, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from .config import (
    Profile,
    delete_profile,
    get_config_path,
    load_config,
    upsert_profile,
)

__all__ = ["initialize", "status", "teardown"]

Backend = Literal["sagemaker", "ray_aws"]
_SUPPORTED_BACKENDS = typing.get_args(Backend)


def initialize(
    *,
    backend: Backend = "sagemaker",
    region: Optional[str] = None,
    stack_name: Optional[str] = None,
    role_arn: Optional[str] = None,
    bucket: Optional[str] = None,
    aws_profile: Optional[str] = None,
    profile_name: str = "default",
) -> None:
    """Provision AG-Cloud resources and save a config profile.

    Either deploy a CloudFormation stack (creates an IAM role + S3 bucket
    from scratch) or persist pre-existing resources — pass ``role_arn`` and
    ``bucket`` together to skip CloudFormation. The persisted profile is the
    canonical record; call :func:`status` to inspect it.

    Parameters
    ----------
    backend
        Which AG-Cloud backend to provision.
    region
        AWS region. Falls back to the boto3-configured region, then ``us-east-1``.
    stack_name
        CloudFormation stack name. Auto-generated as ``ag-cloud-<backend>``
        if not given.
    role_arn, bucket
        If BOTH are given, skip CloudFormation and record these pre-existing
        resources.
    aws_profile
        Local AWS profile name (from ``~/.aws/credentials`` or SSO) to use as
        the base identity. Defaults to the standard boto3 credential chain.
    profile_name
        Name of the AG-Cloud profile entry in ``~/.autogluon/cloud.yaml``.

    Raises
    ------
    ValueError
        If ``role_arn`` and ``bucket`` aren't both provided together, or if
        an unknown backend is passed.
    RuntimeError
        If AWS credentials can't be detected.
    """
    if (role_arn is None) ^ (bucket is None):
        raise ValueError("`role_arn` and `bucket` must be provided together.")
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend {backend!r}. Choose from {_SUPPORTED_BACKENDS}.")

    session = _verified_session(aws_profile=aws_profile, region=region)
    region = session.region_name or "us-east-1"

    if role_arn is None:
        stack_name = stack_name or f"ag-cloud-{backend.replace('_', '-')}"
        print(f"Deploying CloudFormation stack {stack_name!r} (this typically takes ~1 minute)...")
        role_arn, bucket = _provision_stack(session, stack_name=stack_name, backend=backend)
        print(f"Stack {stack_name!r} deployed.")

    profile = Profile(
        region=region,
        role_arn=role_arn,
        bucket=bucket,
        backend=backend,
        stack_name=stack_name,
        aws_profile=aws_profile,
    )
    upsert_profile(profile_name, profile)
    print(f"Saved profile {profile_name!r} to {get_config_path()}")


def status(profile_name: Optional[str] = None, *, check_role: bool = True) -> Dict[str, Any]:
    """Return a health snapshot of the named (or active) profile.

    On found=True the dict contains ``profile`` (Profile), ``is_active`` (bool),
    ``config_path`` (str), and ``checks`` (dict of ``bucket`` / ``stack`` /
    ``role`` strings, each ``"ok"`` or a failure description).

    On found=False the dict contains ``config_path`` and (if a profile name
    was given but missing) ``reason`` and ``available_profiles``.

    Makes real AWS calls. Pass ``check_role=False`` to skip ``iam:GetRole``.
    """
    config = load_config()
    if config is None or not config.profiles:
        return {"found": False, "config_path": str(get_config_path())}

    try:
        profile = config.get_profile(profile_name)
    except KeyError:
        return {
            "found": False,
            "config_path": str(get_config_path()),
            "reason": "profile not found",
            "requested_profile": profile_name,
            "available_profiles": sorted(config.profiles),
        }
    active = profile_name or config.active_profile

    session = _boto_session(aws_profile=profile.aws_profile, region=profile.region)
    checks: Dict[str, str] = {"bucket": _check_bucket(session, profile.bucket)}
    if profile.stack_name:
        checks["stack"] = _check_stack(session, profile.stack_name)
    if check_role:
        checks["role"] = _check_role(session, profile.role_arn)

    return {
        "found": True,
        "config_path": str(get_config_path()),
        "profile_name": active,
        "is_active": active == config.active_profile,
        "profile": profile,
        "checks": checks,
    }


def teardown(
    profile_name: Optional[str] = None,
    *,
    delete_bucket_contents: bool = False,
) -> None:
    """Delete the CloudFormation stack and remove the profile from config.

    Parameters
    ----------
    profile_name
        Profile to tear down. Defaults to the active profile.
    delete_bucket_contents
        Empty the S3 bucket before stack deletion. Required if the bucket is
        non-empty; otherwise CloudFormation will refuse to delete it.

    Raises
    ------
    botocore.exceptions.ClientError, botocore.exceptions.WaiterError
        If stack deletion or bucket emptying fails.
    """
    config = load_config()
    if config is None or not config.profiles:
        print("No AG-Cloud config found — nothing to tear down.")
        return
    try:
        profile = config.get_profile(profile_name)
    except KeyError:
        print(f"Profile {profile_name!r} not found. Available profiles: {sorted(config.profiles)}")
        return
    active = profile_name or config.active_profile

    if profile.stack_name is None:
        delete_profile(active)
        print(f"Removed profile {active!r} (no stack to delete).")
        return

    session = _boto_session(aws_profile=profile.aws_profile, region=profile.region)
    if delete_bucket_contents:
        _empty_bucket(session, profile.bucket)

    print(f"Deleting CloudFormation stack {profile.stack_name!r} (this typically takes ~1 minute)...")
    cfn = session.client("cloudformation")
    cfn.delete_stack(StackName=profile.stack_name)
    cfn.get_waiter("stack_delete_complete").wait(StackName=profile.stack_name)

    delete_profile(active)
    print(f"Stack {profile.stack_name!r} deleted; removed profile {active!r}.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _boto_session(aws_profile: Optional[str] = None, region: Optional[str] = None) -> boto3.Session:
    kwargs: Dict[str, Any] = {}
    if aws_profile:
        kwargs["profile_name"] = aws_profile
    if region:
        kwargs["region_name"] = region
    return boto3.Session(**kwargs)


def _verified_session(aws_profile: Optional[str] = None, region: Optional[str] = None) -> boto3.Session:
    """Build a session and verify it can call sts:GetCallerIdentity."""
    session = _boto_session(aws_profile=aws_profile, region=region)
    try:
        session.client("sts").get_caller_identity()
    except (NoCredentialsError, ClientError, BotoCoreError) as e:
        raise RuntimeError(
            "Could not detect AWS credentials. Run `aws configure`, set AWS_* "
            "env vars, use AWS SSO, or pass `aws_profile=<name>`."
        ) from e
    return session


def _provision_stack(session: boto3.Session, *, stack_name: str, backend: Backend) -> tuple[str, str]:
    """Deploy the bundled CFN template and return ``(role_arn, bucket_name)``."""
    cfn = session.client("cloudformation")
    template = resources.files("autogluon.cloud.templates").joinpath(f"ag_cloud_{backend}.yaml")

    try:
        cfn.create_stack(
            StackName=stack_name,
            TemplateBody=template.read_text(),
            Capabilities=["CAPABILITY_NAMED_IAM"],
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "AlreadyExistsException":
            raise
        print(f"Stack {stack_name!r} already exists — reusing it.")

    cfn.get_waiter("stack_create_complete").wait(StackName=stack_name)

    outputs = {
        o["OutputKey"]: o["OutputValue"]
        for o in cfn.describe_stacks(StackName=stack_name)["Stacks"][0].get("Outputs", [])
    }
    missing = {"RoleARN", "BucketName"} - outputs.keys()
    if missing:
        raise RuntimeError(f"Stack outputs missing required keys: {missing}")
    return outputs["RoleARN"], outputs["BucketName"]


def _check_bucket(session: boto3.Session, bucket: str) -> str:
    try:
        session.client("s3").head_bucket(Bucket=bucket)
        return "ok"
    except ClientError as e:
        return f"failed ({e.response.get('Error', {}).get('Code', '?')})"


def _check_stack(session: boto3.Session, stack_name: str) -> str:
    try:
        return session.client("cloudformation").describe_stacks(StackName=stack_name)["Stacks"][0]["StackStatus"]
    except ClientError as e:
        return e.response["Error"]["Message"]


def _check_role(session: boto3.Session, role_arn: str) -> str:
    """Verify the IAM role exists. Does not assume it — that would bypass the
    caller's boto3 identity.
    """
    try:
        session.client("iam").get_role(RoleName=role_arn.rsplit("/", 1)[-1])
        return "ok"
    except ClientError as e:
        return f"failed ({e.response.get('Error', {}).get('Code', '?')})"


def _empty_bucket(session: boto3.Session, bucket: str) -> None:
    print(f"Emptying bucket {bucket!r}...")
    b = session.resource("s3").Bucket(bucket)
    b.object_versions.delete()
    b.objects.delete()
