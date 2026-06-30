"""Command-line interface for AutoGluon-Cloud setup.

Wraps the four public functions in :mod:`autogluon.cloud`
(``bootstrap``, ``register``, ``status``, ``teardown``) with Click commands +
Rich-based prompts. The Python API is the source of truth for behavior.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional

import boto3
import click
from packaging.version import Version
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from . import bootstrap as _bootstrap
from . import register as _register
from . import status as _status
from . import teardown as _teardown
from .config import load_config
from .version import __version__

# Backends exposed in the CLI. See all supported backends at backend.constant.SUPPORTED_BACKENDS
_CLI_BACKENDS = ("sagemaker",)

_console = Console()


def _template_url(backend: str) -> str:
    """GitHub URL of the CloudFormation template for ``backend``.

    On a stable release the URL points at that release's tag (e.g. ``v0.5.0``); on
    nightly/dev builds it falls back to ``master``.
    """
    v = Version(__version__)
    ref = "master" if (v.is_prerelease or v.is_devrelease) else f"v{v.public}"
    return (
        f"https://github.com/autogluon/autogluon-cloud/blob/{ref}/"
        f"src/autogluon/cloud/templates/ag_cloud_{backend}.yaml"
    )


def _make_session(aws_profile: Optional[str], region: Optional[str]) -> Optional[boto3.Session]:
    """Build a boto3.Session honoring optional --aws-profile and --region. Returns None if neither given."""
    if not aws_profile and not region:
        return None
    kwargs = {}
    if aws_profile:
        kwargs["profile_name"] = aws_profile
    if region:
        kwargs["region_name"] = region
    return boto3.Session(**kwargs)


def _abort_on_error(fn, *args, **kwargs):
    """Run ``fn`` and convert RuntimeError/ValueError into a Click exit (no traceback)."""
    try:
        return fn(*args, **kwargs)
    except (RuntimeError, ValueError) as e:
        raise click.ClickException(str(e)) from e


@contextmanager
def _quiet_logger(name: str, level: int = logging.WARNING):
    """Temporarily raise a logger's level so its INFO output doesn't fight the rich spinner."""
    logger = logging.getLogger(name)
    prior = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(prior)


@click.group()
@click.version_option(package_name="autogluon.cloud")
def cli() -> None:
    """Set up and manage your AWS environment for AutoGluon-Cloud."""


@cli.command()
@click.option("--backend", type=click.Choice(_CLI_BACKENDS), default="sagemaker", show_default=True)
@click.option("--region", default=None, help="AWS region for the stack.")
@click.option("--stack-name", default=None, help="CloudFormation stack name.")
@click.option("--aws-profile", default=None, help="AWS profile from ~/.aws/credentials.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def bootstrap(
    backend: str,
    region: Optional[str],
    stack_name: Optional[str],
    aws_profile: Optional[str],
    yes: bool,
) -> None:
    """One-time setup to run AutoGluon-Cloud on AWS."""
    _console.print("\nOne-time setup to run AutoGluon-Cloud on AWS.\n")

    if region is None:
        detected = boto3.Session(profile_name=aws_profile).region_name if aws_profile else boto3.Session().region_name
        region = Prompt.ask("AWS region", default=detected or "us-east-1")

    session = _make_session(aws_profile, region)
    if stack_name is None:
        default_stack = f"ag-cloud-{backend.replace('_', '-')}"
        stack_name = Prompt.ask("Stack name", default=default_stack)
    effective_stack = stack_name

    _console.print(
        f"This will use CloudFormation to create AWS resources (IAM roles, S3 bucket, etc.) "
        f"needed to run AutoGluon-Cloud with '{backend}'.\n"
        f"Verify the template: {_template_url(backend)}\n"
    )

    plan = Table.grid(padding=(0, 2))
    plan.add_row("[bold]Backend[/bold]", backend)
    plan.add_row("[bold]Region[/bold]", region)
    plan.add_row("[bold]Stack[/bold]", effective_stack)
    if aws_profile:
        plan.add_row("[bold]Profile[/bold]", aws_profile)
    _console.print(Panel(plan, title="Deployment plan", border_style="cyan"))

    if not yes and not Confirm.ask("Proceed?", default=True):
        raise click.Abort()

    with _quiet_logger("autogluon.cloud"), _console.status(f"Deploying stack '{effective_stack}'...", spinner="dots"):
        _abort_on_error(_bootstrap, backend=backend, stack_name=effective_stack, session=session)

    config = load_config()
    if config and backend in config.backends:
        bc = config.backends[backend]
        cfn_url = (
            f"https://{bc.region}.console.aws.amazon.com/cloudformation/home"
            f"?region={bc.region}#/stacks?filteringText={effective_stack}"
        )
        _console.print("\n[bold]Created resources:[/bold]")
        _console.print(f"  Role:   {bc.role_arn}")
        _console.print(f"  Bucket: {bc.bucket}")
        _console.print(f"  Stack:  {cfn_url}\n")


@cli.command()
@click.option("--backend", type=click.Choice(_CLI_BACKENDS), default="sagemaker", show_default=True)
@click.option("--role", default=None, help="IAM role ARN.")
@click.option("--bucket", default=None, help="S3 bucket name.")
@click.option("--region", default=None, help="AWS region for the resources.")
@click.option("--stack-name", default=None, help="CloudFormation stack name (for teardown to find later).")
def register(
    backend: str,
    role: Optional[str],
    bucket: Optional[str],
    region: Optional[str],
    stack_name: Optional[str],
) -> None:
    """Use your own IAM role and S3 bucket with AutoGluon-Cloud."""
    role = role or Prompt.ask("IAM role ARN")
    bucket = bucket or Prompt.ask("S3 bucket name")
    region = region or Prompt.ask("AWS region", default="us-east-1")
    if stack_name is None:
        # Blank → keep None. Only meaningful if you want `teardown` to delete a CFN stack later.
        stack_name = Prompt.ask("CloudFormation stack name (optional, blank to skip)", default="") or None

    _abort_on_error(
        _register,
        role=role,
        bucket=bucket,
        region=region,
        backend=backend,
        stack_name=stack_name,
    )


@cli.command()
@click.option("--region", default=None, help="AWS region.")
@click.option("--aws-profile", default=None, help="AWS profile from ~/.aws/credentials.")
def status(region: Optional[str], aws_profile: Optional[str]) -> None:
    """Check that configured AWS resources exist and are accessible."""
    config = load_config()
    if config is None or not config.backends:
        _console.print(
            "[yellow]No AutoGluon-Cloud config found.[/yellow] Run `autogluon-cloud bootstrap` to create one."
        )
        return

    session = _make_session(aws_profile, region)
    reports = _status(session=session)

    for name, report in reports.items():
        table = Table(title=f"[bold cyan]{name}[/bold cyan]", show_header=False, box=None)
        table.add_row("Region", report.config.region)
        table.add_row("Bucket", report.config.bucket)
        table.add_row("Role", report.config.role_arn)
        if report.config.stack_name:
            table.add_row("Stack", report.config.stack_name)
        table.add_section()
        for check_name, result in report.checks.items():
            healthy = result.startswith("ok") or result.endswith("_COMPLETE")
            color = "green" if healthy else "red"
            table.add_row(check_name, f"[{color}]{result}[/{color}]")
        _console.print(table)


@cli.command()
@click.option(
    "--backend",
    type=click.Choice(_CLI_BACKENDS),
    default=None,
    help="Only tear down this backend (default: all).",
)
@click.option("--region", default=None, help="AWS region.")
@click.option("--aws-profile", default=None, help="AWS profile from ~/.aws/credentials.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def teardown(
    backend: Optional[str],
    region: Optional[str],
    aws_profile: Optional[str],
    yes: bool,
) -> None:
    """Remove AWS resources created by bootstrap."""
    config = load_config()
    if config is None or not config.backends:
        _console.print("[yellow]No AutoGluon-Cloud config found — nothing to tear down.[/yellow]")
        return

    targets = [backend] if backend else sorted(config.backends)
    target_label = backend or "ALL backends: " + ", ".join(targets)

    if not yes:
        # Bucket reminder so the user can `aws s3 rm` before CFN refuses to delete a non-empty bucket.
        buckets = [config.backends[t].bucket for t in targets if t in config.backends]
        warning = Panel(
            f"About to tear down [bold]{target_label}[/bold].\n\n"
            f"CloudFormation will refuse to delete buckets that aren't empty:\n"
            + "\n".join(f"  • s3://{b}" for b in buckets)
            + "\n\nIf they hold data you want to keep, cancel now. Otherwise, empty them first with\n"
            "`aws s3 rm s3://<bucket> --recursive` if you haven't already.",
            title="[red]Teardown confirmation[/red]",
            border_style="red",
        )
        _console.print(warning)
        if not Confirm.ask("Proceed with teardown?", default=False):
            raise click.Abort()

    session = _make_session(aws_profile, region)
    _abort_on_error(_teardown, backend=backend, session=session)


def main() -> None:
    """Entry point for the ``autogluon-cloud`` console script."""
    cli()


if __name__ == "__main__":
    main()
