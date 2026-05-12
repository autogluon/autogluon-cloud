"""Public Python API for AG-Cloud setup — mirrors the `autogluon-cloud` CLI.

Usage from a notebook or script:

    >>> import autogluon.cloud
    >>> autogluon.cloud.init()                 # interactive, prompts inline
    >>> autogluon.cloud.status()               # returns a dict of health checks
    >>> autogluon.cloud.teardown(yes=True)     # non-interactive teardown

All three functions also accept keyword arguments for full non-interactive
use, mirroring the CLI flags.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from . import _setup
from ._util import SUPPORTED_BACKENDS, console, get_template_path
from .config import load_config

__all__ = ["init", "status", "teardown"]


def init(
    *,
    profile_name: str = "default",
    backend: Optional[str] = None,
    region: Optional[str] = None,
    stack_name: Optional[str] = None,
    role_arn: Optional[str] = None,
    bucket: Optional[str] = None,
    aws_profile: Optional[str] = None,
    yes: bool = False,
    interactive: Optional[bool] = None,
) -> Dict[str, Any]:
    """Provision AG-Cloud resources and save a config profile.

    Call this once per AWS account to create the IAM role + S3 bucket via
    CloudFormation, then use :class:`~autogluon.cloud.TabularCloudPredictor`
    and friends with no arguments — they'll auto-load this config.

    Parameters
    ----------
    profile_name:
        Name of the profile entry in ``~/.autogluon/cloud.yaml``.
    backend:
        ``"sagemaker"`` (default) or ``"ray_aws"``. Prompted if ``None`` and
        running interactively.
    region:
        AWS region. Falls back to the configured boto3 region, then prompt.
    stack_name:
        CloudFormation stack name. Auto-generated as ``ag-cloud-<backend>``
        if not given.
    role_arn, bucket:
        If BOTH are provided, skip CloudFormation entirely and save a config
        pointing at those pre-existing resources. Useful in enterprises with
        centrally-managed IAM.
    aws_profile:
        Name of a local AWS profile (from ``~/.aws/credentials`` or SSO) to
        use as the base identity for provisioning and subsequent role
        assumption.
    yes:
        Skip all confirmation prompts.
    interactive:
        If ``True``, ask for missing values via prompts (works in notebooks
        and terminals). If ``False``, fail on missing required values.
        If ``None`` (default), autodetect: interactive if stdin is a tty or
        we're in Jupyter.

    Returns
    -------
    A dict with ``profile_name``, ``region``, ``bucket``, ``role_arn``,
    ``stack_name`` (``None`` if user supplied existing resources).
    """
    if (role_arn is None) ^ (bucket is None):
        raise ValueError("`role_arn` and `bucket` must be provided together.")

    if interactive is None:
        interactive = _is_interactive()

    console.print(Panel.fit("[bold]AutoGluon-Cloud Setup[/bold]", border_style="cyan"))

    resolved = _setup.resolve_defaults(
        backend=backend,
        region=region,
        stack_name=stack_name,
        aws_profile=aws_profile,
    )
    identity = resolved["identity"]
    _setup.print_identity_banner(identity, aws_profile)

    backend = backend or (
        Prompt.ask("Which backend?", choices=list(SUPPORTED_BACKENDS), default="sagemaker")
        if interactive
        else resolved["backend"]
    )
    if region is None:
        if identity["region"]:
            region = identity["region"]
        elif interactive:
            region = Prompt.ask("AWS region", default="us-east-1")
        else:
            region = resolved["region"]

    existing = load_config()
    if existing and profile_name in existing.profiles and not yes:
        if interactive:
            if not Confirm.ask(
                f"Profile [bold]{profile_name}[/bold] already exists — overwrite?",
                default=False,
            ):
                console.print("Aborted.")
                return {"cancelled": True}
        else:
            raise RuntimeError(f"Profile {profile_name!r} already exists. Pass yes=True to overwrite.")

    if role_arn and bucket:
        profile = _setup.save_existing_resources_profile(
            profile_name=profile_name,
            backend=backend,
            region=region,
            role_arn=role_arn,
            bucket=bucket,
            aws_profile=aws_profile,
        )
        _setup.print_success(profile_name, profile)
        return _profile_to_result(profile_name, profile)

    stack_name = resolved["stack_name"]

    _setup.print_plan(
        backend=backend,
        region=region,
        stack_name=stack_name,
        account_id=identity["account"],
        template_path=get_template_path(backend),
    )
    if not yes:
        if interactive:
            if not Confirm.ask("Proceed with deployment?", default=True):
                console.print("Aborted.")
                return {"cancelled": True}
        # Non-interactive without yes=True: refuse to deploy.
        else:
            raise RuntimeError("Refusing to deploy without confirmation. Pass yes=True to proceed.")

    outputs = _setup.provision_stack(
        stack_name=stack_name,
        backend=backend,
        region=region,
        aws_profile=aws_profile,
    )
    profile = _setup.save_provisioned_profile(
        profile_name=profile_name,
        backend=backend,
        region=region,
        stack_name=stack_name,
        outputs=outputs,
        aws_profile=aws_profile,
    )
    _setup.print_success(profile_name, profile)
    return _profile_to_result(profile_name, profile)


def status(
    profile_name: Optional[str] = None,
    *,
    check_role: bool = True,
) -> Dict[str, Any]:
    """Return a structured health snapshot of the named (or active) profile.

    Returns a dict with ``found`` (bool). When ``found`` is True, also has:
    ``profile_name``, ``is_active``, ``profile`` (a :class:`Profile`),
    ``checks`` (dict of ``bucket`` / ``stack`` / ``role`` strings).

    When no config exists, prints a one-line hint suggesting
    ``autogluon.cloud.init()`` and still returns ``{"found": False, ...}``
    so callers can branch on the result.
    """
    snapshot = _setup.gather_status(profile_name=profile_name, check_role=check_role)
    if not snapshot["found"]:
        if snapshot.get("reason") == "profile not found":
            console.print(
                f"[red]Profile {snapshot['requested_profile']!r} not found.[/red] "
                f"Available: {snapshot['available_profiles']}"
            )
        else:
            console.print(
                f"[yellow]No AG-Cloud config found at {snapshot['config_path']}.[/yellow] "
                "Run [bold]autogluon.cloud.init()[/bold] to get started."
            )
    return snapshot


def teardown(
    profile_name: Optional[str] = None,
    *,
    delete_bucket_contents: bool = False,
    yes: bool = False,
    interactive: Optional[bool] = None,
) -> Dict[str, Any]:
    """Delete the CloudFormation stack and remove the profile from config.

    Parameters
    ----------
    profile_name:
        Profile to tear down. Defaults to the active profile.
    delete_bucket_contents:
        Empty the S3 bucket first. Required if the bucket is non-empty,
        otherwise CFN will fail to delete it.
    yes:
        Skip the typed-confirmation prompt.
    interactive:
        If ``True``, ask the user to confirm by typing the stack name. If
        ``False`` and ``yes`` is not set, raises ``RuntimeError`` — destructive
        actions never run silently by default.
    """
    if interactive is None:
        interactive = _is_interactive()

    config = load_config()
    if config is None or not config.profiles:
        console.print("[yellow]No config to tear down.[/yellow]")
        return {"removed": False, "reason": "no config"}
    try:
        profile = config.get_profile(profile_name)
    except KeyError:
        console.print(f"[red]Profile {profile_name!r} not found.[/red] Available: {sorted(config.profiles)}")
        return {
            "removed": False,
            "reason": "profile not found",
            "requested_profile": profile_name,
            "available_profiles": sorted(config.profiles),
        }
    active = profile_name or config.active_profile

    if profile.stack_name and not yes:
        if interactive:
            confirmation = Prompt.ask(f"Type the stack name [bold]{profile.stack_name}[/bold] to confirm")
            if confirmation != profile.stack_name:
                console.print("[yellow]Confirmation did not match. Aborted.[/yellow]")
                return {"removed": False, "reason": "not confirmed"}
        else:
            raise RuntimeError(
                f"Refusing to tear down {profile.stack_name!r} without confirmation. Pass yes=True to proceed."
            )

    result = _setup.teardown_profile(
        profile_name=active,
        delete_bucket_contents=delete_bucket_contents,
    )
    if result["removed"] and result.get("stack_deleted"):
        console.print(f"[green]Teardown complete. Profile {active} removed from config.[/green]")
    elif result["removed"]:
        console.print(f"[yellow]Profile had no associated stack. Removed {active} from config.[/yellow]")
    return result


def _is_interactive() -> bool:
    try:
        import sys

        if sys.stdin.isatty():
            return True
    except Exception:
        pass
    try:
        from IPython import get_ipython  # type: ignore

        return get_ipython() is not None
    except Exception:
        return False


def _profile_to_result(profile_name: str, profile) -> Dict[str, Any]:
    return {
        "profile_name": profile_name,
        "backend": profile.backend,
        "region": profile.region,
        "bucket": profile.bucket,
        "role_arn": profile.role_arn,
        "stack_name": profile.stack_name,
    }
