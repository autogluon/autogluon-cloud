"""Python API for provisioning AutoGluon-Cloud on AWS.

Usage::

    from autogluon.cloud import bootstrap, register, status, teardown

    bootstrap()                                          # deploy CFN + save config
    register(backend=, role=, bucket=, region=)          # save existing resources
    status()                                             # dict of StatusReport per backend
    teardown()                                           # delete CFN + config (all backends)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from typing import Dict, Literal, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from .backend.constant import SUPPORTED_BACKENDS
from .config import (
    BackendConfig,
    CloudConfig,
    delete_config,
    get_config_path,
    load_config,
    save_config,
)

__all__ = ["bootstrap", "register", "status", "teardown", "StatusReport"]


@dataclass
class StatusReport:
    """Health snapshot for a single backend."""

    config: BackendConfig
    config_path: str
    checks: Dict[str, str] = field(default_factory=dict)


# Keep these values in sync with SUPPORTED_BACKENDS in backend/constant.py.
BackendName = Literal["sagemaker", "ray_aws"]


def bootstrap(
    *,
    backend: BackendName = "sagemaker",
    stack_name: Optional[str] = None,
    session: Optional[boto3.Session] = None,
    verbose: bool = True,
) -> None:
    """Deploy the CloudFormation stack and persist resource identifiers.

    On completion the IAM role and S3 bucket created by the stack are saved to ``~/.autogluon/cloud.yaml`` via
    :func:`register`. If you already have an IAM role and bucket in place, call :func:`register` directly and
    skip this function entirely.

    Each backend has its own slot in the config file, so calling :func:`bootstrap` for ``sagemaker`` and again
    for ``ray_aws`` keeps both in the config.

    Parameters
    ----------
    backend
        Which AutoGluon-Cloud backend to provision.
    stack_name
        CloudFormation stack name. Auto-generated as ``ag-cloud-<backend>`` if not given.
    session
        A ``boto3.Session`` to use for AWS calls. If ``None``, a default session is constructed from the standard
        credential chain (env vars, ``~/.aws/credentials``, SSO, instance profile).
    verbose
        If ``True`` (default), print progress messages to stdout.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend {backend!r}. Choose from {SUPPORTED_BACKENDS}.")

    session, account = _verified_session(session)
    region = session.region_name
    if region is None:
        raise RuntimeError(
            "AWS region not configured. Set AWS_DEFAULT_REGION, run `aws configure`, "
            "or pass `session=boto3.Session(region_name=...)`."
        )
    stack_name = stack_name or f"ag-cloud-{backend.replace('_', '-')}"

    if verbose:
        print(f"Deploying CloudFormation stack {stack_name!r} (account {account}, region {region}, ~1 minute)...")
    role_arn, bucket = _provision_stack(session, stack_name=stack_name, backend=backend)
    if verbose:
        print(f"Stack {stack_name!r} deployed.")

    register(
        role=role_arn,
        bucket=bucket,
        region=region,
        backend=backend,
        stack_name=stack_name,
        verbose=verbose,
    )


def register(
    *,
    role: str,
    bucket: str,
    region: str,
    backend: BackendName = "sagemaker",
    stack_name: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """Persist resource identifiers to ``~/.autogluon/cloud.yaml`` under the given backend key.

    Use this when you already have an IAM role and S3 bucket — for example, centrally provisioned by your platform
    team — and just want AutoGluon-Cloud to remember them.

    If a config entry already exists for ``backend``, it is overwritten. Other backends in the file are left
    untouched.

    Parameters
    ----------
    role
        ARN of an IAM role suitable for SageMaker / Ray to assume. Named ``role`` for consistency with the SageMaker
        Python SDK (which uses ``role`` as the parameter name).
    bucket
        S3 bucket name where AutoGluon-Cloud will read/write artifacts.
    region
        AWS region for AutoGluon-Cloud operations.
    backend
        Which AutoGluon-Cloud backend the resources are intended for. Selects the slot in ``cloud.yaml``.
    stack_name
        Optional CloudFormation stack name. If you deployed the resources via your own CFN stack and want
        :func:`teardown` to be able to delete it later, pass the name here. Defaults to ``None``, meaning teardown
        will only remove the config entry, not touch AWS.
    verbose
        If ``True`` (default), print progress messages to stdout.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend {backend!r}. Choose from {SUPPORTED_BACKENDS}.")
    config = load_config() or CloudConfig()
    config.backends[backend] = BackendConfig(
        region=region,
        role_arn=role,
        bucket=bucket,
        stack_name=stack_name,
    )
    save_config(config)
    if verbose:
        print(f"Saved AutoGluon-Cloud config for backend {backend!r} to {get_config_path()}")


def status(
    *,
    session: Optional[boto3.Session] = None,
) -> Dict[str, StatusReport]:
    """Return health snapshots keyed by backend name, one per configured backend.

    Each :class:`StatusReport` has:

    * ``config`` — the saved :class:`BackendConfig`
    * ``config_path`` — path to ``~/.autogluon/cloud.yaml``
    * ``checks`` — dict of ``bucket`` / ``stack`` / ``role`` to a status string. ``"ok"`` means the resource exists;
      ``"ok (unverified)"`` means the caller lacks the IAM permission to verify and the resource may still be fine;
      anything else is a failure description.

    Returns an empty dict if no config exists. Makes real AWS calls. Pass ``session=`` to use specific credentials;
    otherwise the standard boto3 credential chain is used (with the saved region as a default).
    """
    config = load_config()
    if config is None:
        return {}

    reports: Dict[str, StatusReport] = {}
    for name, backend_config in config.backends.items():
        sess = session or boto3.Session(region_name=backend_config.region)
        checks: Dict[str, str] = {"bucket": _check_bucket(sess, backend_config.bucket)}
        if backend_config.stack_name:
            checks["stack"] = _check_stack(sess, backend_config.stack_name)
        checks["role"] = _check_role(sess, backend_config.role_arn)
        reports[name] = StatusReport(
            config=backend_config,
            config_path=str(get_config_path()),
            checks=checks,
        )
    return reports


def teardown(
    *,
    backend: Optional[BackendName] = None,
    session: Optional[boto3.Session] = None,
) -> None:
    """Delete a backend's CloudFormation stack and remove its config entry.

    With ``backend=None`` (default), tears down every configured backend and removes the config file. With
    ``backend="sagemaker"``, tears down just that one and leaves any other backends in the config.

    The S3 buckets are **not** emptied for you: CloudFormation refuses to delete a non-empty bucket, so you must
    remove their contents (e.g. via ``aws s3 rm s3://<bucket> --recursive``) before calling :func:`teardown`. This
    is intentional — buckets may contain training artifacts or model weights that are expensive to recreate.

    Parameters
    ----------
    backend
        Which backend to tear down. ``None`` (default) tears down all configured backends.
    session
        A ``boto3.Session`` to use for AWS calls. If ``None``, a default session is built from the standard
        credential chain, with each backend's saved region applied automatically.
    """
    config = load_config()
    if config is None or not config.backends:
        print("No AutoGluon-Cloud config found — nothing to tear down.")
        return

    if backend is not None and backend not in config.backends:
        print(f"Backend {backend!r} not in config. Available: {sorted(config.backends)}")
        return

    targets = [backend] if backend is not None else list(config.backends)
    for name in targets:
        backend_config = config.backends[name]
        if backend_config.stack_name is None:
            print(f"[{name}] no stack to delete.")
        else:
            sess, account = _verified_session(session or boto3.Session(region_name=backend_config.region))
            print(
                f"[{name}] Deleting CloudFormation stack {backend_config.stack_name!r} "
                f"(account {account}, region {backend_config.region}, ~1 minute)..."
            )
            cfn = sess.client("cloudformation")
            cfn.delete_stack(StackName=backend_config.stack_name)
            cfn.get_waiter("stack_delete_complete").wait(StackName=backend_config.stack_name)
            print(f"[{name}] Stack {backend_config.stack_name!r} deleted.")
        del config.backends[name]

    if config.backends:
        save_config(config)
        print(f"Removed {targets} from config; remaining backends: {sorted(config.backends)}.")
    else:
        delete_config()
        print("Removed config file.")


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

    stack_existed = False
    try:
        cfn.create_stack(
            StackName=stack_name,
            TemplateBody=template.read_text(),
            Capabilities=["CAPABILITY_NAMED_IAM"],
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "AlreadyExistsException":
            raise
        stack_existed = True
        print(f"Stack {stack_name!r} already exists — reusing it.")

    if not stack_existed:
        cfn.get_waiter("stack_create_complete").wait(StackName=stack_name)

    desc = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]
    outputs = {o["OutputKey"]: o["OutputValue"] for o in desc.get("Outputs", [])}
    missing = {"RoleARN", "BucketName"} - outputs.keys()
    if missing:
        raise RuntimeError(
            f"Stack {stack_name!r} is in {desc['StackStatus']} and missing required outputs: {missing}. "
            f"Delete it via the CloudFormation console and re-run."
        )
    return outputs["RoleARN"], outputs["BucketName"]


def _is_permission_error(e: ClientError) -> bool:
    # AWS error codes that mean "the caller lacks IAM permission to read this", as
    # distinct from "the resource doesn't exist". We surface these as ``"unverified"``
    # rather than ``"failed"`` so users don't think their setup is broken when really
    # it's just a permissions gap on the side of whoever is running ``status()``.
    return e.response.get("Error", {}).get("Code", "") in {
        "AccessDenied",
        "AccessDeniedException",
        "Forbidden",
        "UnauthorizedOperation",
    }


def _check_bucket(session: boto3.Session, bucket: str) -> str:
    try:
        session.client("s3").head_bucket(Bucket=bucket)
        return "ok"
    except ClientError as e:
        if _is_permission_error(e):
            return "ok (unverified — caller lacks s3:HeadBucket)"
        return f"failed ({e.response.get('Error', {}).get('Code', '?')})"


def _check_stack(session: boto3.Session, stack_name: str) -> str:
    try:
        return session.client("cloudformation").describe_stacks(StackName=stack_name)["Stacks"][0]["StackStatus"]
    except ClientError as e:
        if _is_permission_error(e):
            return "ok (unverified — caller lacks cloudformation:DescribeStacks)"
        return e.response["Error"]["Message"]


def _check_role(session: boto3.Session, role_arn: str) -> str:
    """Verify the IAM role exists via iam:GetRole. Doesn't call sts:AssumeRole —
    we only check existence, not the caller's permission to assume it.
    """
    # iam:GetRole's RoleName takes the bare name, not the path. For a role with a path
    # (e.g. arn:aws:iam::123:role/prod/MyRole), the name is the segment after final '/'
    role_name = role_arn.rsplit("/", 1)[-1]
    try:
        session.client("iam").get_role(RoleName=role_name)
        return "ok"
    except ClientError as e:
        if _is_permission_error(e):
            return "ok (unverified — caller lacks iam:GetRole)"
        return f"failed ({e.response.get('Error', {}).get('Code', '?')})"
