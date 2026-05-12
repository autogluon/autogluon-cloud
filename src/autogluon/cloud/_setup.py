"""Shared provisioning logic for AG-Cloud — used by both the CLI and the
top-level Python API (`autogluon.cloud.init/status/teardown`).

The functions here deliberately don't do any argument prompting. Prompting
lives at the edges (CLI command handler, `init()` wrapper) so the same core
can be driven non-interactively from code or interactively from a shell/
notebook.
"""

from __future__ import annotations

import time
from importlib.abc import Traversable
from typing import Any, Dict, Optional

from botocore.exceptions import ClientError
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ._util import (
    SUPPORTED_BACKENDS,
    console,
    detect_aws_identity,
    format_identity,
    get_template_path,
    make_boto_session,
)
from .config import (
    Profile,
    delete_profile,
    get_config_path,
    load_config,
    upsert_profile,
)

TERMINAL_CREATE_STATUSES = {
    "CREATE_COMPLETE",
    "CREATE_FAILED",
    "ROLLBACK_COMPLETE",
    "ROLLBACK_FAILED",
}
TERMINAL_DELETE_STATUSES = {"DELETE_COMPLETE", "DELETE_FAILED"}


class SetupError(RuntimeError):
    """Raised when setup / teardown can't continue."""


def resolve_defaults(
    backend: Optional[str],
    region: Optional[str],
    stack_name: Optional[str],
    aws_profile: Optional[str],
) -> Dict[str, Any]:
    """Fill in sensible defaults and detect AWS identity.

    Returns a dict with ``backend``, ``region``, ``stack_name``, and
    ``identity`` (the STS caller identity). Raises ``SetupError`` if creds
    can't be found.
    """
    identity = detect_aws_identity(region=region, aws_profile=aws_profile)
    if identity is None:
        raise SetupError(
            "Could not detect AWS credentials. Run `aws configure`, set "
            "AWS_* env vars, use AWS SSO, or pass `aws_profile=<name>`."
        )

    backend = backend or "sagemaker"
    if backend not in SUPPORTED_BACKENDS:
        raise SetupError(f"Unsupported backend {backend!r}. Choose from {SUPPORTED_BACKENDS}.")
    region = region or identity["region"] or "us-east-1"
    stack_name = stack_name or f"ag-cloud-{backend.replace('_', '-')}"
    return {
        "backend": backend,
        "region": region,
        "stack_name": stack_name,
        "identity": identity,
    }


def save_existing_resources_profile(
    *,
    profile_name: str,
    backend: str,
    region: str,
    role_arn: str,
    bucket: str,
    aws_profile: Optional[str] = None,
) -> Profile:
    """Persist a config profile pointing at pre-existing AWS resources."""
    profile = Profile(
        region=region,
        role_arn=role_arn,
        bucket=bucket,
        backend=backend,
        stack_name=None,
        aws_profile=aws_profile,
    )
    upsert_profile(profile_name, profile)
    return profile


def provision_stack(
    *,
    stack_name: str,
    backend: str,
    region: str,
    aws_profile: Optional[str] = None,
    show_progress: bool = True,
) -> Dict[str, str]:
    """Deploy the bundled CloudFormation template and return its outputs.

    Outputs always contain ``RoleARN`` and ``BucketName``.
    """
    template_path = get_template_path(backend)
    session = make_boto_session(aws_profile=aws_profile, region=region)
    cfn = session.client("cloudformation")
    template_body = template_path.read_text()

    if show_progress:
        console.print(f"Deploying CloudFormation stack [bold]{stack_name}[/bold]...")

    try:
        cfn.create_stack(
            StackName=stack_name,
            TemplateBody=template_body,
            Capabilities=["CAPABILITY_NAMED_IAM"],
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "AlreadyExistsException":
            if show_progress:
                console.print(f"[yellow]Stack {stack_name} already exists — reusing it.[/yellow]")
        else:
            raise SetupError(f"Failed to create stack: {e}") from e

    final_status = _wait_for_status(
        cfn,
        stack_name,
        TERMINAL_CREATE_STATUSES,
        show_progress=show_progress,
    )
    if final_status != "CREATE_COMPLETE":
        raise SetupError(f"Stack ended in status {final_status}. Check the CloudFormation console.")
    if show_progress:
        console.print(f"[green]Stack {stack_name} deployed successfully.[/green]")
    return _get_stack_outputs(cfn, stack_name)


def save_provisioned_profile(
    *,
    profile_name: str,
    backend: str,
    region: str,
    stack_name: str,
    outputs: Dict[str, str],
    aws_profile: Optional[str] = None,
) -> Profile:
    profile = Profile(
        region=region,
        role_arn=outputs["RoleARN"],
        bucket=outputs["BucketName"],
        backend=backend,
        stack_name=stack_name,
        aws_profile=aws_profile,
    )
    upsert_profile(profile_name, profile)
    return profile


def gather_status(profile_name: Optional[str] = None, check_role: bool = True) -> Dict[str, Any]:
    """Return a structured snapshot of the named (or active) profile."""
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

    base = make_boto_session(aws_profile=profile.aws_profile, region=profile.region)
    checks: Dict[str, str] = {"bucket": _check_bucket(base, profile.bucket)}
    if profile.stack_name:
        checks["stack"] = _check_stack(base, profile.stack_name)
    if check_role:
        checks["role"] = _check_role(profile)

    return {
        "found": True,
        "config_path": str(get_config_path()),
        "profile_name": active,
        "is_active": active == config.active_profile,
        "profile": profile,
        "checks": checks,
    }


def teardown_profile(
    profile_name: Optional[str] = None,
    *,
    delete_bucket_contents: bool = False,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Delete the CFN stack for a profile and remove it from config."""
    config = load_config()
    if config is None or not config.profiles:
        return {"removed": False, "reason": "no config"}
    try:
        profile = config.get_profile(profile_name)
    except KeyError:
        return {
            "removed": False,
            "reason": "profile not found",
            "requested_profile": profile_name,
            "available_profiles": sorted(config.profiles),
        }
    active = profile_name or config.active_profile

    if profile.stack_name is None:
        delete_profile(active)
        return {"removed": True, "stack_deleted": False, "profile": active}

    base = make_boto_session(aws_profile=profile.aws_profile, region=profile.region)
    if delete_bucket_contents:
        _empty_bucket(base, profile.bucket, show_progress=show_progress)

    cfn = base.client("cloudformation")
    try:
        cfn.delete_stack(StackName=profile.stack_name)
    except ClientError as e:
        raise SetupError(f"Failed to start stack deletion: {e}") from e

    final_status = _wait_for_status(
        cfn,
        profile.stack_name,
        TERMINAL_DELETE_STATUSES,
        show_progress=show_progress,
    )
    if final_status == "DELETE_FAILED":
        raise SetupError("Stack deletion failed. Check the CloudFormation console.")
    delete_profile(active)
    return {"removed": True, "stack_deleted": True, "profile": active}


def print_plan(*, backend: str, region: str, stack_name: str, account_id: str, template_path: Traversable) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[bold]Account[/bold]", account_id)
    table.add_row("[bold]Region[/bold]", region)
    table.add_row("[bold]Backend[/bold]", backend)
    table.add_row("[bold]Stack name[/bold]", stack_name)
    table.add_row("[bold]Template[/bold]", str(template_path))
    console.print(Panel(table, title="Deployment plan", border_style="yellow"))


def print_identity_banner(identity: Dict[str, Any], aws_profile: Optional[str]) -> None:
    profile_hint = f" (aws profile: [bold]{aws_profile}[/bold])" if aws_profile else ""
    console.print(f"[green]AWS credentials detected[/green] — {format_identity(identity)}{profile_hint}")


def print_success(profile_name: str, profile: Profile) -> None:
    config_path = get_config_path()
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[bold]Profile[/bold]", profile_name)
    table.add_row("[bold]Region[/bold]", profile.region)
    table.add_row("[bold]Bucket[/bold]", profile.bucket)
    table.add_row("[bold]Role ARN[/bold]", profile.role_arn)
    if profile.stack_name:
        table.add_row("[bold]Stack[/bold]", profile.stack_name)
    console.print(Panel(table, title=f"Saved to {config_path}", border_style="green"))
    console.print(
        "\n[bold]You're ready![/bold] Try:\n"
        "  [cyan]>>> from autogluon.cloud import TabularCloudPredictor[/cyan]\n"
        "  [cyan]>>> predictor = TabularCloudPredictor()[/cyan]\n"
    )


def _wait_for_status(cfn, stack_name: str, terminal: set, show_progress: bool = True) -> str:
    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Waiting for stack...", total=None)
            return _poll_until_terminal(cfn, stack_name, terminal, progress=progress, task_id=task)
    return _poll_until_terminal(cfn, stack_name, terminal)


def _poll_until_terminal(cfn, stack_name: str, terminal: set, *, progress=None, task_id=None) -> str:
    while True:
        try:
            desc = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]
            status = desc["StackStatus"]
        except ClientError as e:
            # describe_stacks raises once the stack is fully gone.
            if "does not exist" in str(e):
                return "DELETE_COMPLETE"
            raise
        if progress is not None:
            progress.update(task_id, description=f"Stack status: {status}")
        if status in terminal:
            return status
        time.sleep(5)


def _get_stack_outputs(cfn, stack_name: str) -> Dict[str, str]:
    desc = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]
    outputs = {o["OutputKey"]: o["OutputValue"] for o in desc.get("Outputs", [])}
    required = {"RoleARN", "BucketName"}
    missing = required - outputs.keys()
    if missing:
        raise SetupError(f"Stack outputs missing required keys: {missing}")
    return outputs


def _check_bucket(session, bucket: str) -> str:
    try:
        session.client("s3").head_bucket(Bucket=bucket)
        return "ok"
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "?")
        return f"failed ({code})"
    except Exception as e:  # noqa: BLE001 — diagnostic summary
        return f"error: {e}"


def _check_stack(session, stack_name: str) -> str:
    try:
        cfn = session.client("cloudformation")
        desc = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]
        return desc["StackStatus"]
    except ClientError as e:
        return e.response["Error"]["Message"]


def _check_role(profile: Profile) -> str:
    try:
        base = make_boto_session(aws_profile=profile.aws_profile, region=profile.region)
        base.client("sts").assume_role(
            RoleArn=profile.role_arn,
            RoleSessionName="ag-cloud-status-check",
            DurationSeconds=900,
        )
        return "ok"
    except Exception as e:  # noqa: BLE001 — diagnostic summary
        return f"{type(e).__name__}: {e}"


def _empty_bucket(session, bucket: str, show_progress: bool = True) -> None:
    if show_progress:
        console.print(f"Emptying bucket [bold]{bucket}[/bold]...")
    s3 = session.resource("s3")
    b = s3.Bucket(bucket)
    try:
        b.object_versions.delete()
        b.objects.delete()
    except ClientError as e:
        raise SetupError(f"Failed to empty bucket: {e}") from e
