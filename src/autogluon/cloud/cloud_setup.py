"""Python API for provisioning AutoGluon-Cloud on AWS.

Usage::

    from autogluon.cloud import bootstrap, register, status, teardown

    bootstrap()                                          # deploy CFN + save config
    register(backend=, role=, bucket=, region=)          # save existing resources
    status()                                             # dict of StatusReport per backend
    teardown()                                           # delete CFN + config (all backends)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib import resources
from typing import Literal

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

logger = logging.getLogger(__name__)


@dataclass
class StatusReport:
    """Health snapshot for a single backend."""

    config: BackendConfig
    config_path: str
    checks: dict[str, str] = field(default_factory=dict)


# Keep these values in sync with SUPPORTED_BACKENDS in backend/constant.py.
BackendName = Literal["sagemaker"]


def bootstrap(
    *,
    backend: BackendName = "sagemaker",
    stack_name: str | None = None,
    session: boto3.Session | None = None,
) -> None:
    """Create the IAM role and S3 bucket used by AutoGluon-Cloud and save them to ``~/.autogluon/cloud.yaml``.

    Deploys a CloudFormation stack (or reuses an existing stack with the same name) and saves its outputs via
    :func:`register`. If you already have an IAM role and bucket, call :func:`register` instead.

    Parameters
    ----------
    backend: BackendName, default = "sagemaker"
        Which AutoGluon-Cloud backend to provision.
    stack_name: str | None, default = None
        CloudFormation stack name. Defaults to ``ag-cloud-<backend>``.
    session: boto3.Session | None, default = None
        Session used for AWS calls; its region is where the resources are created. If ``None``, uses the default
        boto3 credential chain and region.
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

    logger.info(f"Deploying CloudFormation stack {stack_name!r} (account {account}, region {region}, ~1 minute)...")
    role_arn, bucket = _provision_stack(session, stack_name=stack_name, backend=backend)
    logger.info(f"Stack {stack_name!r} deployed.")

    register(
        role=role_arn,
        bucket=bucket,
        region=region,
        backend=backend,
        stack_name=stack_name,
        session=session,
    )


def register(
    *,
    role: str,
    bucket: str,
    region: str,
    backend: BackendName = "sagemaker",
    stack_name: str | None = None,
    session: boto3.Session | None = None,
) -> None:
    """Save an existing IAM role and S3 bucket to ``~/.autogluon/cloud.yaml``.

    Use this instead of :func:`bootstrap` if the resources already exist, e.g. provisioned by your platform team.
    Overwrites any existing entry for ``backend``.

    Parameters
    ----------
    role: str
        ARN of the SageMaker execution role.
    bucket: str
        Name of the S3 bucket for artifacts, without ``s3://`` or a prefix. Must be in ``region``.
    region: str
        AWS region where jobs and endpoints run.
    backend: BackendName, default = "sagemaker"
        Backend the resources are used for.
    stack_name: str | None, default = None
        CloudFormation stack that owns the resources. If set, :func:`teardown` deletes this stack; otherwise it
        only removes the config entry.
    session: boto3.Session | None, default = None
        Session used to verify the bucket region. If ``None``, uses the default boto3 session.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend {backend!r}. Choose from {SUPPORTED_BACKENDS}.")
    bucket = bucket.removeprefix("s3://").rstrip("/")
    if "/" in bucket:
        raise ValueError(
            f"`bucket` must be a bare bucket name without prefixes (got {bucket!r}). "
            "Pass prefixes via `cloud_output_path=` on the predictor/model instead."
        )
    _validate_bucket_region(session=session or boto3.Session(), bucket=bucket, region=region)
    config = load_config() or CloudConfig()
    config.backends[backend] = BackendConfig(
        region=region,
        role_arn=role,
        bucket=bucket,
        stack_name=stack_name,
    )
    save_config(config)
    logger.info(f"Saved AutoGluon-Cloud config for backend {backend!r} to {get_config_path()}")


def status(
    *,
    session: boto3.Session | None = None,
) -> dict[str, StatusReport]:
    """Check that the resources saved in ``~/.autogluon/cloud.yaml`` exist.

    Parameters
    ----------
    session: boto3.Session | None, default = None
        Session used for AWS calls. If ``None``, uses the default boto3 credentials with each backend's saved
        region.

    Returns
    -------
    dict[str, StatusReport]
        One report per configured backend, keyed by backend name; empty if no config exists. Each report has:

        * ``config``: the saved backend config (region, role ARN, bucket, and stack name).
        * ``config_path``: path to the config file.
        * ``checks``: status of the ``"bucket"``, ``"role"``, and (if ``stack_name`` is set) ``"stack"``. The
          bucket and role checks return ``"ok"``, and the stack check returns its CloudFormation status, e.g.
          ``"CREATE_COMPLETE"``. ``"ok (unverified ...)"`` means the caller lacks permission to check the resource.
          Anything else describes a failure.
    """
    config = load_config()
    if config is None:
        return {}

    reports: dict[str, StatusReport] = {}
    for name, backend_config in config.backends.items():
        sess = session or boto3.Session(region_name=backend_config.region)
        checks: dict[str, str] = {"bucket": _check_bucket(sess, backend_config.bucket)}
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
    backend: BackendName | None = None,
    session: boto3.Session | None = None,
) -> None:
    """Delete the CloudFormation stack created by :func:`bootstrap` and remove the config entry.

    For backends saved with :func:`register` without a ``stack_name``, only the config entry is removed.

    The S3 bucket is **not** emptied for you, and CloudFormation can't delete a non-empty bucket. Empty it first,
    e.g. with ``aws s3 rm s3://<bucket> --recursive``.

    Parameters
    ----------
    backend: BackendName | None, default = None
        Backend to tear down. If ``None``, tears down all configured backends and deletes the config file.
    session: boto3.Session | None, default = None
        Session used for AWS calls. If ``None``, uses the default boto3 credentials with each backend's saved
        region.
    """
    config = load_config()
    if config is None or not config.backends:
        logger.warning("No AutoGluon-Cloud config found — nothing to tear down.")
        return

    if backend is not None and backend not in config.backends:
        logger.warning(f"Backend {backend!r} not in config. Available: {sorted(config.backends)}")
        return

    targets = [backend] if backend is not None else list(config.backends)
    for name in targets:
        backend_config = config.backends[name]
        if backend_config.stack_name is None:
            logger.info(f"[{name}] no stack to delete.")
        else:
            sess, account = _verified_session(session or boto3.Session(region_name=backend_config.region))
            logger.info(
                f"[{name}] Deleting CloudFormation stack {backend_config.stack_name!r} "
                f"(account {account}, region {backend_config.region}, ~1 minute)..."
            )
            cfn = sess.client("cloudformation")
            cfn.delete_stack(StackName=backend_config.stack_name)
            cfn.get_waiter("stack_delete_complete").wait(StackName=backend_config.stack_name)
            logger.info(f"[{name}] Stack {backend_config.stack_name!r} deleted.")
        del config.backends[name]

    if config.backends:
        save_config(config)
        logger.info(f"Removed {targets} from config; remaining backends: {sorted(config.backends)}.")
    else:
        delete_config()
        logger.info("Removed config file.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _verified_session(session: boto3.Session | None) -> tuple[boto3.Session, str]:
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
        logger.warning(f"Stack {stack_name!r} already exists — reusing it.")

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


def _validate_bucket_region(*, session: boto3.Session, bucket: str, region: str) -> None:
    """Raise if the bucket is in a different region than ``region``. Silently skips if the bucket
    region can't be determined (missing bucket, network issues, etc.).

    Cross-region ``head_bucket`` calls return 403 even when the caller lacks ``s3:HeadBucket``
    permission, but the response still carries the ``x-amz-bucket-region`` header — so we read it
    from the error path too, otherwise the very mismatch this function exists to catch slips through
    whenever the caller's role is locked down.
    """
    try:
        response = session.client("s3").head_bucket(Bucket=bucket)
        bucket_region = response["ResponseMetadata"]["HTTPHeaders"].get("x-amz-bucket-region")
    except ClientError as e:
        bucket_region = e.response.get("ResponseMetadata", {}).get("HTTPHeaders", {}).get("x-amz-bucket-region")
    except BotoCoreError:
        return
    if not bucket_region:
        return
    if bucket_region != region:
        raise ValueError(
            f"Bucket {bucket!r} is in region {bucket_region!r}, but you registered it under {region!r}. "
            "SageMaker requires the bucket and the job region to match. Either pass `--region "
            f"{bucket_region}` (and run jobs there), or pick a bucket in {region!r}."
        )


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
