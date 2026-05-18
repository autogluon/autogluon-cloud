"""Python API for provisioning AutoGluon-Cloud on AWS.

Usage::

    from autogluon.cloud import bootstrap, register, status, teardown

    bootstrap()                                          # deploy CFN + save config
    register(role_arn=..., bucket=..., region=...)       # save existing resources
    status()                                             # dict of health checks
    teardown(delete_bucket_contents=True)                # delete CFN + config
"""

from __future__ import annotations

from importlib import resources
from typing import Any, Dict, Literal, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from .backend.constant import RAY_AWS, SAGEMAKER, SUPPORTED_SETUP_BACKENDS
from .config import (
    CloudConfig,
    delete_config,
    get_config_path,
    load_config,
    save_config,
)

__all__ = ["bootstrap", "register", "status", "teardown"]

BackendName = Literal[SAGEMAKER, RAY_AWS]


def bootstrap(
    *,
    backend: BackendName = "sagemaker",
    stack_name: Optional[str] = None,
    session: Optional[boto3.Session] = None,
) -> None:
    """Deploy the CloudFormation stack and persist resource identifiers.

    On completion the IAM role and S3 bucket created by the stack are saved
    to ``~/.autogluon/cloud.yaml`` via :func:`register`. If you
    already have an IAM role and bucket in place, call :func:`register`
    directly and skip this function entirely.

    Parameters
    ----------
    backend
        Which AG-Cloud backend to provision.
    stack_name
        CloudFormation stack name. Auto-generated as ``ag-cloud-<backend>``
        if not given.
    session
        A ``boto3.Session`` to use for AWS calls. If ``None``, a default
        session is constructed from the standard credential chain
        (env vars, ``~/.aws/credentials``, SSO, instance profile).

    Raises
    ------
    ValueError
        If an unknown backend is passed.
    RuntimeError
        If AWS credentials can't be detected.
    """
    if backend not in SUPPORTED_SETUP_BACKENDS:
        raise ValueError(f"Unsupported backend {backend!r}. Choose from {SUPPORTED_SETUP_BACKENDS}.")

    session, account = _verified_session(session)
    region = session.region_name
    if region is None:
        raise RuntimeError(
            "AWS region not configured. Set AWS_DEFAULT_REGION, run `aws configure`, "
            "or pass `session=boto3.Session(region_name=...)`."
        )
    stack_name = stack_name or f"ag-cloud-{backend.replace('_', '-')}"

    print(f"Deploying CloudFormation stack {stack_name!r} (account {account}, region {region}, ~1 minute)...")
    role_arn, bucket = _provision_stack(session, stack_name=stack_name, backend=backend)
    print(f"Stack {stack_name!r} deployed.")

    register(
        role_arn=role_arn,
        bucket=bucket,
        region=region,
        backend=backend,
        stack_name=stack_name,
    )


def register(
    *,
    role_arn: str,
    bucket: str,
    region: str,
    backend: BackendName = "sagemaker",
    stack_name: Optional[str] = None,
) -> None:
    """Persist resource identifiers to ``~/.autogluon/cloud.yaml``.

    Use this when you already have an IAM role and S3 bucket — for example,
    centrally provisioned by your platform team — and just want AG-Cloud to
    remember them. Pure file I/O; no AWS calls.

    Parameters
    ----------
    role_arn
        ARN of an IAM role suitable for SageMaker / Ray to assume.
    bucket
        S3 bucket name where AG-Cloud will read/write artifacts.
    region
        AWS region for AG-Cloud operations.
    backend
        Which AG-Cloud backend the resources are intended for.
    stack_name
        Optional CloudFormation stack name. If you deployed the resources
        via your own CFN stack and want :func:`teardown` to be able to
        delete it later, pass the name here. Defaults to ``None``, meaning
        teardown will only remove the config file, not touch AWS.

    Raises
    ------
    ValueError
        If an unknown backend is passed.
    """
    if backend not in SUPPORTED_SETUP_BACKENDS:
        raise ValueError(f"Unsupported backend {backend!r}. Choose from {SUPPORTED_SETUP_BACKENDS}.")
    config = CloudConfig(
        region=region,
        role_arn=role_arn,
        bucket=bucket,
        backend=backend,
        stack_name=stack_name,
    )
    save_config(config)
    print(f"Saved AG-Cloud config to {get_config_path()}")


def status(
    *,
    session: Optional[boto3.Session] = None,
    check_role: bool = True,
) -> Dict[str, Any]:
    """Return a health snapshot of the persisted config.

    On found=True the dict contains ``config`` (CloudConfig), ``config_path``
    (str), and ``checks`` (dict of ``bucket`` / ``stack`` / ``role`` strings,
    each ``"ok"`` or a failure description).

    On found=False the dict contains just ``config_path``.

    Makes real AWS calls. Pass ``check_role=False`` to skip ``iam:GetRole``.
    Pass ``session=`` to use specific credentials; otherwise the standard
    boto3 credential chain is used (with the saved region as a default).
    """
    config = load_config()
    if config is None:
        return {"found": False, "config_path": str(get_config_path())}

    session = session or boto3.Session(region_name=config.region)
    checks: Dict[str, str] = {"bucket": _check_bucket(session, config.bucket)}
    if config.stack_name:
        checks["stack"] = _check_stack(session, config.stack_name)
    if check_role:
        checks["role"] = _check_role(session, config.role_arn)

    return {
        "found": True,
        "config_path": str(get_config_path()),
        "config": config,
        "checks": checks,
    }


def teardown(
    *,
    session: Optional[boto3.Session] = None,
    delete_bucket_contents: bool = False,
) -> None:
    """Delete the CloudFormation stack (if any) and remove the config file.

    Parameters
    ----------
    session
        A ``boto3.Session`` to use for AWS calls. If ``None``, a default
        session is built from the standard credential chain, with the saved
        region applied automatically.
    delete_bucket_contents
        Empty the S3 bucket before stack deletion. Required if the bucket is
        non-empty; otherwise CloudFormation will refuse to delete it.

    Raises
    ------
    botocore.exceptions.ClientError, botocore.exceptions.WaiterError
        If stack deletion or bucket emptying fails.
    """
    config = load_config()
    if config is None:
        print("No AG-Cloud config found — nothing to tear down.")
        return

    if config.stack_name is None:
        delete_config()
        print("Removed config (no stack to delete).")
        return

    session, account = _verified_session(session or boto3.Session(region_name=config.region))
    if delete_bucket_contents:
        _empty_bucket(session, config.bucket)

    print(
        f"Deleting CloudFormation stack {config.stack_name!r} "
        f"(account {account}, region {config.region}, ~1 minute)..."
    )
    cfn = session.client("cloudformation")
    cfn.delete_stack(StackName=config.stack_name)
    cfn.get_waiter("stack_delete_complete").wait(StackName=config.stack_name)

    delete_config()
    print(f"Stack {config.stack_name!r} deleted; config removed.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _verified_session(session: Optional[boto3.Session]) -> tuple[boto3.Session, str]:
    """Build a default session if none given and verify it can call STS.

    Returns the session paired with the AWS account ID, so callers can show
    the user what's about to happen (and where) without a second STS call.
    """
    session = session or boto3.Session()
    try:
        identity = session.client("sts").get_caller_identity()
    except (NoCredentialsError, ClientError, BotoCoreError) as e:
        raise RuntimeError(
            "Could not detect AWS credentials. Run `aws configure`, set AWS_* "
            "env vars, use AWS SSO, or pass a configured `boto3.Session`."
        ) from e
    return session, identity["Account"]


def _provision_stack(session: boto3.Session, *, stack_name: str, backend: BackendName) -> tuple[str, str]:
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
    """Verify the IAM role exists via iam:GetRole. Doesn't call sts:AssumeRole —
    we only check existence, not the caller's permission to assume it.
    """
    try:
        session.client("iam").get_role(RoleName=role_arn.rsplit("/", 1)[-1])
        return "ok"
    except ClientError as e:
        return f"failed ({e.response.get('Error', {}).get('Code', '?')})"


def _empty_bucket(session: boto3.Session, bucket: str) -> None:
    print(f"Emptying bucket {bucket!r}...")
    b = session.resource("s3").Bucket(bucket)
    # Both calls are needed for versioned buckets (our CFN template enables
    # versioning): object_versions covers all historical versions + delete
    # markers; objects covers anything left in the current listing.
    b.object_versions.delete()
    b.objects.delete()
