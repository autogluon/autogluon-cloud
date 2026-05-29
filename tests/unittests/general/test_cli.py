"""Tests for the ``autogluon-cloud`` CLI.

We exercise the click commands via ``CliRunner`` and stub the underlying Python API functions, so these tests
verify only the CLI's prompting + argument-passing logic — not the AWS-side behavior (which is covered in
``test_cloud_setup.py``).
"""

import pytest
from click.testing import CliRunner

from autogluon.cloud.cli import cli
from autogluon.cloud.config import CONFIG_DIR_ENV, BackendConfig, CloudConfig, save_config


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    yield tmp_path


@pytest.fixture
def runner():
    return CliRunner()


def _stub_python_api(monkeypatch, *, calls):
    """Replace the four CLI-imported functions with stubs that record their kwargs."""
    monkeypatch.setattr(
        "autogluon.cloud.cli._bootstrap",
        lambda **kw: calls.setdefault("bootstrap", []).append(kw),
    )
    monkeypatch.setattr(
        "autogluon.cloud.cli._register",
        lambda **kw: calls.setdefault("register", []).append(kw),
    )
    monkeypatch.setattr(
        "autogluon.cloud.cli._teardown",
        lambda **kw: calls.setdefault("teardown", []).append(kw),
    )


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------


def test_help_lists_all_commands(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for cmd in ("bootstrap", "register", "status", "teardown"):
        assert cmd in result.output


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_with_yes_skips_confirm_and_calls_python_api(runner, monkeypatch):
    calls = {}
    _stub_python_api(monkeypatch, calls=calls)

    result = runner.invoke(
        cli,
        ["bootstrap", "--backend", "sagemaker", "--region", "us-east-1", "--stack-name", "my-stack", "--yes"],
    )
    assert result.exit_code == 0, result.output
    assert calls["bootstrap"] == [
        {
            "backend": "sagemaker",
            "stack_name": "my-stack",
            "session": pytest.approx(calls["bootstrap"][0]["session"]),
        }
    ]


def test_bootstrap_aborts_when_user_says_no(runner, monkeypatch):
    calls = {}
    _stub_python_api(monkeypatch, calls=calls)

    # First prompt is stack name (accept default), then `n` for Confirm → click.Abort.
    result = runner.invoke(cli, ["bootstrap", "--backend", "sagemaker", "--region", "us-east-1"], input="\nn\n")
    assert result.exit_code != 0
    assert "bootstrap" not in calls  # function was never called


def test_bootstrap_defaults_backend_to_sagemaker(runner, monkeypatch):
    calls = {}
    _stub_python_api(monkeypatch, calls=calls)

    # No --backend flag: defaults to sagemaker without prompting; only the stack-name prompt remains.
    result = runner.invoke(cli, ["bootstrap", "--region", "us-east-1", "--yes"], input="\n")
    assert result.exit_code == 0, result.output
    assert calls["bootstrap"][0]["backend"] == "sagemaker"


def test_bootstrap_prompts_for_missing_region(runner, monkeypatch):
    calls = {}
    _stub_python_api(monkeypatch, calls=calls)

    # Prompts: region ('eu-west-2'), stack name (accept default).
    result = runner.invoke(
        cli,
        ["bootstrap", "--backend", "sagemaker", "--yes"],
        input="eu-west-2\n\n",
    )
    assert result.exit_code == 0, result.output
    # The CLI threads region into the boto3.Session; verify by checking session.region_name.
    assert calls["bootstrap"][0]["session"].region_name == "eu-west-2"


def test_bootstrap_translates_runtime_error_to_clean_exit(runner, monkeypatch):
    def raises(**kw):
        raise RuntimeError("AWS credentials missing")

    monkeypatch.setattr("autogluon.cloud.cli._bootstrap", raises)

    result = runner.invoke(
        cli, ["bootstrap", "--backend", "sagemaker", "--region", "us-east-1", "--stack-name", "s", "--yes"]
    )
    assert result.exit_code != 0
    assert "AWS credentials missing" in result.output
    # ClickException prints the message without a Python traceback.
    assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def test_register_with_all_flags(runner, monkeypatch):
    calls = {}
    _stub_python_api(monkeypatch, calls=calls)

    result = runner.invoke(
        cli,
        [
            "register",
            "--role",
            "arn:aws:iam::123:role/x",
            "--bucket",
            "my-bucket",
            "--region",
            "us-east-1",
            "--stack-name",
            "my-stack",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls["register"][0] == {
        "role": "arn:aws:iam::123:role/x",
        "bucket": "my-bucket",
        "region": "us-east-1",
        "backend": "sagemaker",
        "stack_name": "my-stack",
    }


def test_register_prompts_for_missing_required(runner, monkeypatch):
    calls = {}
    _stub_python_api(monkeypatch, calls=calls)

    # Four prompts: role, bucket, region (default us-east-1), stack_name (blank → None).
    result = runner.invoke(
        cli,
        ["register"],
        input="arn:aws:iam::123:role/x\nmy-bucket\n\n\n",
    )
    assert result.exit_code == 0, result.output
    rk = calls["register"][0]
    assert rk["role"] == "arn:aws:iam::123:role/x"
    assert rk["bucket"] == "my-bucket"
    assert rk["region"] == "us-east-1"
    assert rk["stack_name"] is None


def test_register_prompts_for_stack_name_when_provided(runner, monkeypatch):
    calls = {}
    _stub_python_api(monkeypatch, calls=calls)

    # Same flow, but answer the stack_name prompt with a non-empty value.
    result = runner.invoke(
        cli,
        ["register"],
        input="arn:aws:iam::123:role/x\nmy-bucket\nus-east-1\nmy-stack\n",
    )
    assert result.exit_code == 0, result.output
    assert calls["register"][0]["stack_name"] == "my-stack"


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_without_config_prints_hint(runner):
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "No AutoGluon-Cloud config" in result.output


def test_status_with_config_renders_table(runner, monkeypatch):
    save_config(
        CloudConfig(
            backends={
                "sagemaker": BackendConfig(
                    region="us-east-1",
                    role_arn="arn:aws:iam::123:role/r",
                    bucket="b1",
                    stack_name="ag-cloud-sagemaker",
                ),
            }
        )
    )

    # Stub the underlying python-api status() so we don't hit AWS.
    from autogluon.cloud.cli import _status as real_status  # noqa: F401

    class FakeReport:
        def __init__(self, cfg):
            self.config = cfg
            self.checks = {"bucket": "ok", "role": "ok", "stack": "CREATE_COMPLETE"}

    def fake_status(*, session=None):
        cfg = BackendConfig(
            region="us-east-1",
            role_arn="arn:aws:iam::123:role/r",
            bucket="b1",
            stack_name="ag-cloud-sagemaker",
        )
        return {"sagemaker": FakeReport(cfg)}

    monkeypatch.setattr("autogluon.cloud.cli._status", fake_status)

    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    assert "sagemaker" in result.output
    assert "ag-cloud-sagemaker" in result.output
    assert "CREATE_COMPLETE" in result.output


# ---------------------------------------------------------------------------
# teardown
# ---------------------------------------------------------------------------


def test_teardown_without_config_prints_hint(runner):
    result = runner.invoke(cli, ["teardown"])
    assert result.exit_code == 0
    assert "nothing to tear down" in result.output


def test_teardown_with_yes_skips_confirm_and_calls_python_api(runner, monkeypatch):
    save_config(
        CloudConfig(
            backends={
                "sagemaker": BackendConfig(
                    region="us-east-1",
                    role_arn="arn:...",
                    bucket="b1",
                    stack_name="ag-cloud-sagemaker",
                )
            }
        )
    )
    calls = {}
    _stub_python_api(monkeypatch, calls=calls)

    result = runner.invoke(cli, ["teardown", "--yes"])
    assert result.exit_code == 0, result.output
    assert calls["teardown"][0]["backend"] is None  # no --backend → all


def test_teardown_with_explicit_backend(runner, monkeypatch):
    save_config(
        CloudConfig(
            backends={
                "sagemaker": BackendConfig(
                    region="us-east-1",
                    role_arn="arn:...",
                    bucket="b1",
                    stack_name="s",
                ),
                "ray_aws": BackendConfig(
                    region="us-east-1",
                    role_arn="arn:...",
                    bucket="b2",
                    stack_name="r",
                ),
            }
        )
    )
    calls = {}
    _stub_python_api(monkeypatch, calls=calls)

    result = runner.invoke(cli, ["teardown", "--backend", "sagemaker", "--yes"])
    assert result.exit_code == 0, result.output
    assert calls["teardown"][0]["backend"] == "sagemaker"


def test_teardown_aborts_on_no_confirmation(runner, monkeypatch):
    save_config(
        CloudConfig(
            backends={
                "sagemaker": BackendConfig(
                    region="us-east-1",
                    role_arn="arn:...",
                    bucket="b1",
                    stack_name="s",
                )
            }
        )
    )
    calls = {}
    _stub_python_api(monkeypatch, calls=calls)

    # Default is N for teardown's confirm; an empty answer accepts the default → abort.
    result = runner.invoke(cli, ["teardown"], input="\n")
    assert result.exit_code != 0
    assert "teardown" not in calls
