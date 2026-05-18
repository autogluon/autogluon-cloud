"""Tests for the ``autogluon.cloud`` setup API."""

import pytest

from autogluon.cloud import bootstrap, register, status, teardown
from autogluon.cloud.config import CONFIG_DIR_ENV, CloudConfig, load_config, save_config


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    yield tmp_path


def _register_default():
    register(
        role_arn="arn:aws:iam::111122223333:role/x",
        bucket="b1",
        region="us-east-1",
    )


# ---------------------------------------------------------------------------
# register (no AWS calls — pure file I/O)
# ---------------------------------------------------------------------------


def test_register_writes_file(capsys):
    _register_default()

    config = load_config()
    assert config.role_arn == "arn:aws:iam::111122223333:role/x"
    assert config.bucket == "b1"
    assert config.region == "us-east-1"
    assert config.backend == "sagemaker"
    assert config.stack_name is None
    assert "Saved AG-Cloud config" in capsys.readouterr().out


def test_register_overwrites_existing():
    _register_default()
    register(
        role_arn="arn:aws:iam::111122223333:role/y",
        bucket="b2",
        region="us-west-2",
    )
    config = load_config()
    assert config.bucket == "b2"
    assert config.region == "us-west-2"


def test_register_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        register(
            role_arn="arn:...",
            bucket="b",
            region="us-east-1",
            backend="not-a-backend",
        )


def test_register_records_stack_name_when_given():
    """Users who deployed their own stack can record the name so teardown can clean up."""
    register(
        role_arn="arn:aws:iam::111122223333:role/x",
        bucket="b1",
        region="us-east-1",
        stack_name="my-stack",
    )
    assert load_config().stack_name == "my-stack"


# ---------------------------------------------------------------------------
# bootstrap (CFN deploy + register)
# ---------------------------------------------------------------------------


def test_bootstrap_calls_cfn_then_registers(monkeypatch, capsys):
    """bootstrap should deploy a stack and persist outputs via register."""

    class FakeSession:
        region_name = "us-east-1"

        def client(self, service):
            raise AssertionError("session.client unused; _provision_stack is stubbed")

    monkeypatch.setattr(
        "autogluon.cloud.cloud_setup._verified_session",
        lambda s: (FakeSession(), "123456789012"),
    )
    monkeypatch.setattr(
        "autogluon.cloud.cloud_setup._provision_stack",
        lambda session, stack_name, backend: ("arn:aws:iam::123:role/r", "ag-cloud-bucket"),
    )

    bootstrap(backend="sagemaker", stack_name="my-stack")

    config = load_config()
    assert config.role_arn == "arn:aws:iam::123:role/r"
    assert config.bucket == "ag-cloud-bucket"
    assert config.stack_name == "my-stack"
    assert config.region == "us-east-1"

    out = capsys.readouterr().out
    assert "Deploying CloudFormation stack 'my-stack'" in out
    assert "account 123456789012" in out
    assert "region us-east-1" in out
    assert "deployed" in out
    assert "Saved AG-Cloud config" in out


def test_bootstrap_returns_none(monkeypatch):
    monkeypatch.setattr(
        "autogluon.cloud.cloud_setup._verified_session",
        lambda s: (type("S", (), {"region_name": "us-east-1"})(), "123456789012"),
    )
    monkeypatch.setattr(
        "autogluon.cloud.cloud_setup._provision_stack",
        lambda session, stack_name, backend: ("arn:...", "b"),
    )
    assert bootstrap() is None


def test_bootstrap_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        bootstrap(backend="not-a-backend")


def test_bootstrap_default_stack_name_uses_backend(monkeypatch):
    captured = {}

    class FakeSession:
        region_name = "us-east-1"

    def fake_provision(session, stack_name, backend):
        captured["stack_name"] = stack_name
        return "arn:...", "b"

    monkeypatch.setattr(
        "autogluon.cloud.cloud_setup._verified_session",
        lambda s: (FakeSession(), "123456789012"),
    )
    monkeypatch.setattr("autogluon.cloud.cloud_setup._provision_stack", fake_provision)

    bootstrap(backend="ray_aws")
    assert captured["stack_name"] == "ag-cloud-ray-aws"


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_without_config_returns_not_found():
    result = status()
    assert result["found"] is False


def test_status_with_config(monkeypatch):
    _register_default()
    monkeypatch.setattr("autogluon.cloud.cloud_setup._check_bucket", lambda s, b: "ok")
    monkeypatch.setattr("autogluon.cloud.cloud_setup._check_role", lambda s, r: "ok")

    result = status()
    assert result["found"] is True
    assert isinstance(result["config"], CloudConfig)
    assert result["checks"] == {"bucket": "ok", "role": "ok"}


def test_status_check_role_false_skips_iam(monkeypatch):
    _register_default()
    monkeypatch.setattr("autogluon.cloud.cloud_setup._check_bucket", lambda s, b: "ok")

    result = status(check_role=False)
    assert "role" not in result["checks"]


def test_status_includes_stack_check_when_stack_name_set(monkeypatch):
    register(role_arn="arn:...", bucket="b", region="us-east-1", stack_name="s")
    monkeypatch.setattr("autogluon.cloud.cloud_setup._check_bucket", lambda s, b: "ok")
    monkeypatch.setattr("autogluon.cloud.cloud_setup._check_role", lambda s, r: "ok")
    monkeypatch.setattr(
        "autogluon.cloud.cloud_setup._check_stack",
        lambda s, n: "CREATE_COMPLETE",
    )

    result = status()
    assert result["checks"]["stack"] == "CREATE_COMPLETE"


# ---------------------------------------------------------------------------
# teardown
# ---------------------------------------------------------------------------


def test_teardown_without_config_is_noop(capsys):
    assert teardown() is None
    assert "nothing to tear down" in capsys.readouterr().out


def test_teardown_no_stack_just_removes_config(capsys):
    _register_default()
    teardown()
    assert load_config() is None
    assert "no stack to delete" in capsys.readouterr().out


def test_teardown_with_stack_deletes_stack(monkeypatch):
    save_config(
        CloudConfig(
            region="us-east-1",
            role_arn="arn:...",
            bucket="b",
            backend="sagemaker",
            stack_name="ag-cloud-sagemaker",
        )
    )
    calls = []

    class FakeWaiter:
        def wait(self, **kw):
            pass

    class FakeCFN:
        def delete_stack(self, StackName):
            calls.append(StackName)

        def get_waiter(self, name):
            assert name == "stack_delete_complete"
            return FakeWaiter()

    class FakeSTS:
        def get_caller_identity(self):
            return {"Account": "123456789012"}

    class FakeSession:
        def client(self, service):
            if service == "cloudformation":
                return FakeCFN()
            if service == "sts":
                return FakeSTS()
            raise AssertionError(f"unexpected client: {service}")

        def resource(self, service):
            raise AssertionError("delete_bucket_contents=False; resource() not expected")

    teardown(session=FakeSession())
    assert calls == ["ag-cloud-sagemaker"]
    assert load_config() is None
