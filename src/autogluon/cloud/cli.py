"""
Command-line interface for AutoGluon-Cloud setup.

Wraps the four public functions in :mod:`autogluon.cloud`
(``bootstrap``, ``register``, ``status``, ``teardown``) with Click commands +
Rich-based prompts. The Python API is the source of truth for behavior;
this module only handles user interaction (prompting for missing
values, formatting output, confirming destructive operations) and translates
Python exceptions into clean CLI exit codes.
"""

from __future__ import annotations

from typing import Optional

import boto3
import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from . import bootstrap as _bootstrap
from . import register as _register
from . import status as _status
from . import teardown as _teardown
from .backend.constant import SUPPORTED_BACKENDS
from .config import load_config

_console = Console()


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


@click.group()
@click.version_option(package_name="autogluon.cloud")
def cli() -> None:
    """Manage AutoGluon-Cloud setup on AWS."""


@cli.command()
@click.option(
    "--backend",
    type=click.Choice(SUPPORTED_BACKENDS),
    default=None,
    help="Backend to provision. Prompted if not given.",
)
@click.option("--region", default=None, help="AWS region. Falls back to your boto3 default.")
@click.option("--stack-name", default=None, help="CFN stack name (auto-generated if not given).")
@click.option("--aws-profile", default=None, help="Named AWS profile from ~/.aws/credentials.")
@click.option("--yes", "-y", is_flag=True, help="Skip the deployment confirmation prompt.")
def bootstrap(
    backend: Optional[str],
    region: Optional[str],
    stack_name: Optional[str],
    aws_profile: Optional[str],
    yes: bool,
) -> None:
    """Deploy the CloudFormation stack and persist the config."""
    if backend is None:
        backend = Prompt.ask(
            "Which backend?",
            choices=list(SUPPORTED_BACKENDS),
            default="sagemaker",
        )

    if region is None:
        detected = boto3.Session(profile_name=aws_profile).region_name if aws_profile else boto3.Session().region_name
        region = Prompt.ask("AWS region", default=detected or "us-east-1")

    session = _make_session(aws_profile, region)
    effective_stack = stack_name or f"ag-cloud-{backend.replace('_', '-')}"

    plan = Table.grid(padding=(0, 2))
    plan.add_row("[bold]Backend[/bold]", backend)
    plan.add_row("[bold]Region[/bold]", region)
    plan.add_row("[bold]Stack name[/bold]", effective_stack)
    if aws_profile:
        plan.add_row("[bold]AWS profile[/bold]", aws_profile)
    _console.print(Panel(plan, title="Deployment plan", border_style="cyan"))

    if not yes and not Confirm.ask("Proceed?", default=True):
        raise click.Abort()

    _abort_on_error(
        _bootstrap,
        backend=backend,
        stack_name=stack_name,
        session=session,
    )


@cli.command()
@click.option(
    "--backend",
    type=click.Choice(SUPPORTED_BACKENDS),
    default="sagemaker",
    show_default=True,
)
@click.option("--role", default=None, help="IAM role ARN. Prompted if not given.")
@click.option("--bucket", default=None, help="S3 bucket name. Prompted if not given.")
@click.option("--region", default=None, help="AWS region for the resources. Prompted if not given.")
@click.option("--stack-name", default=None, help="Optional CFN stack name to remember for teardown.")
def register(
    backend: str,
    role: Optional[str],
    bucket: Optional[str],
    region: Optional[str],
    stack_name: Optional[str],
) -> None:
    """Persist an existing IAM role + S3 bucket as the config for a backend (no AWS calls)."""
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
@click.option("--region", default=None, help="Override the saved region for the AWS calls.")
@click.option("--aws-profile", default=None, help="Named AWS profile from ~/.aws/credentials.")
def status(region: Optional[str], aws_profile: Optional[str]) -> None:
    """Show a per-backend health snapshot of the saved config."""
    config = load_config()
    if config is None or not config.backends:
        _console.print("[yellow]No AG-Cloud config found.[/yellow] Run `autogluon-cloud bootstrap` to create one.")
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
            color = "green" if result.startswith("ok") else "red"
            table.add_row(check_name, f"[{color}]{result}[/{color}]")
        _console.print(table)


@cli.command()
@click.option(
    "--backend",
    type=click.Choice(SUPPORTED_BACKENDS),
    default=None,
    help="Tear down just this backend. Defaults to all configured backends.",
)
@click.option("--region", default=None, help="Override the saved region for the AWS calls.")
@click.option("--aws-profile", default=None, help="Named AWS profile from ~/.aws/credentials.")
@click.option("--yes", "-y", is_flag=True, help="Skip the destructive-action confirmation.")
def teardown(
    backend: Optional[str],
    region: Optional[str],
    aws_profile: Optional[str],
    yes: bool,
) -> None:
    """Delete CloudFormation stack(s) and remove the config entry/file."""
    config = load_config()
    if config is None or not config.backends:
        _console.print("[yellow]No AG-Cloud config found — nothing to tear down.[/yellow]")
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
