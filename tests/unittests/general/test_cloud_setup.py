"""Tests for the ``autogluon.cloud`` setup API."""

import pytest

from autogluon.cloud import bootstrap, register, status, teardown
from autogluon.cloud.config import (
    CONFIG_DIR_ENV,
    BackendConfig,
    CloudConfig,
    load_config,
    save_config,
)


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    yield tmp_path


def _register_default(backend="sagemaker"):
    register(
        role="arn:aws:iam::111122223333:role/x",
        bucket="b1",
        region="us-east-1",
        backend=backend,
    )


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def test_register_writes_file(capsys):
    _register_default()

    cfg = load_config()
    assert "sagemaker" in cfg.backends
    sage = cfg.backends["sagemaker"]
    assert sage.role_arn == "arn:aws:iam::111122223333:role/x"
    assert sage.bucket == "b1"
    assert sage.region == "us-east-1"
    assert sage.stack_name is None
    assert "Saved AG-Cloud config for backend 'sagemaker'" in capsys.readouterr().out


def test_register_overwrites_same_backend():
    _register_default()
    register(
        role="arn:aws:iam::111122223333:role/y",
        bucket="b2",
        region="us-west-2",
    )
    cfg = load_config()
    assert cfg.backends["sagemaker"].bucket == "b2"
    assert cfg.backends["sagemaker"].region == "us-west-2"


def test_register_keeps_other_backends_untouched():
    """Adding ray_aws shouldn't disturb sagemaker."""
    _register_default(backend="sagemaker")
    register(
        role="arn:aws:iam::111122223333:role/r",
        bucket="ray-bucket",
        region="us-east-1",
        backend="ray_aws",
    )
    cfg = load_config()
    assert set(cfg.backends) == {"sagemaker", "ray_aws"}
    assert cfg.backends["sagemaker"].bucket == "b1"
    assert cfg.backends["ray_aws"].bucket == "ray-bucket"


def test_register_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        register(
            role="arn:...",
            bucket="b",
            region="us-east-1",
            backend="not-a-backend",
        )


def test_register_records_stack_name_when_given():
    register(
        role="arn:aws:iam::111122223333:role/x",
        bucket="b1",
        region="us-east-1",
        stack_name="my-stack",
    )
    assert load_config().backends["sagemaker"].stack_name == "my-stack"


# ---------------------------------------------------------------------------
# bootstrap (CFN deploy + register)
# ---------------------------------------------------------------------------


def test_bootstrap_calls_cfn_then_registers(monkeypatch, capsys):
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

    cfg = load_config()
    assert cfg.backends["sagemaker"].role_arn == "arn:aws:iam::123:role/r"
    assert cfg.backends["sagemaker"].bucket == "ag-cloud-bucket"
    assert cfg.backends["sagemaker"].stack_name == "my-stack"

    out = capsys.readouterr().out
    assert "Deploying CloudFormation stack 'my-stack'" in out
    assert "account 123456789012" in out
    assert "deployed" in out


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


def test_status_without_config_returns_empty_dict():
    assert status() == {}


def test_status_returns_one_per_backend(monkeypatch):
    _register_default(backend="sagemaker")
    register(
        role="arn:...",
        bucket="ray-bucket",
        region="us-east-1",
        backend="ray_aws",
    )
    monkeypatch.setattr("autogluon.cloud.cloud_setup._check_bucket", lambda s, b: "ok")
    monkeypatch.setattr("autogluon.cloud.cloud_setup._check_role", lambda s, r: "ok")

    reports = status()
    assert set(reports) == {"sagemaker", "ray_aws"}
    assert isinstance(reports["sagemaker"].config, BackendConfig)
    assert reports["sagemaker"].checks == {"bucket": "ok", "role": "ok"}
    assert reports["ray_aws"].config.bucket == "ray-bucket"


def test_status_includes_stack_check_when_stack_name_set(monkeypatch):
    register(
        role="arn:...",
        bucket="b",
        region="us-east-1",
        stack_name="s",
    )
    monkeypatch.setattr("autogluon.cloud.cloud_setup._check_bucket", lambda s, b: "ok")
    monkeypatch.setattr("autogluon.cloud.cloud_setup._check_role", lambda s, r: "ok")
    monkeypatch.setattr(
        "autogluon.cloud.cloud_setup._check_stack",
        lambda s, n: "CREATE_COMPLETE",
    )

    reports = status()
    assert reports["sagemaker"].checks["stack"] == "CREATE_COMPLETE"


def test_check_role_returns_unverified_on_access_denied():
    """AccessDenied means the caller lacks permission to verify, not that the role is broken."""
    from botocore.exceptions import ClientError

    from autogluon.cloud.cloud_setup import _check_role

    class FakeIAM:
        def get_role(self, RoleName):
            raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no perms"}}, "GetRole")

    class FakeSession:
        def client(self, service):
            assert service == "iam"
            return FakeIAM()

    result = _check_role(FakeSession(), "arn:aws:iam::123:role/x")
    assert "unverified" in result


# ---------------------------------------------------------------------------
# teardown
# ---------------------------------------------------------------------------


def test_teardown_without_config_is_noop(capsys):
    assert teardown() is None
    assert "nothing to tear down" in capsys.readouterr().out


def test_teardown_no_stacks_just_removes_config(capsys):
    """Backends registered without stack_name → only the config is removed."""
    _register_default()
    teardown()
    assert load_config() is None
    out = capsys.readouterr().out
    assert "no stack to delete" in out
    assert "Removed config" in out


def test_teardown_with_stack_deletes_each_backend(monkeypatch):
    """Multiple backends with stacks → all get deleted, then config removed."""
    save_config(
        CloudConfig(
            backends={
                "sagemaker": BackendConfig(
                    region="us-east-1",
                    role_arn="arn:...",
                    bucket="b1",
                    stack_name="ag-cloud-sagemaker",
                ),
                "ray_aws": BackendConfig(
                    region="us-east-1",
                    role_arn="arn:...",
                    bucket="b2",
                    stack_name="ag-cloud-ray-aws",
                ),
            }
        )
    )
    deleted = []

    class FakeWaiter:
        def wait(self, **kw):
            pass

    class FakeCFN:
        def delete_stack(self, StackName):
            deleted.append(StackName)

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

    teardown(session=FakeSession())
    assert sorted(deleted) == ["ag-cloud-ray-aws", "ag-cloud-sagemaker"]
    assert load_config() is None


def test_teardown_specific_backend_keeps_others():
    """teardown(backend='sagemaker') removes only that entry; ray_aws stays."""
    _register_default(backend="sagemaker")
    register(role="arn:...", bucket="ray-bucket", region="us-east-1", backend="ray_aws")

    teardown(backend="sagemaker")  # neither has a stack_name → no AWS calls

    cfg = load_config()
    assert cfg is not None
    assert set(cfg.backends) == {"ray_aws"}


def test_teardown_unknown_backend_is_friendly(capsys):
    _register_default(backend="sagemaker")
    teardown(backend="ray_aws")  # not registered
    out = capsys.readouterr().out
    assert "not in config" in out
    assert load_config() is not None  # nothing was removed
