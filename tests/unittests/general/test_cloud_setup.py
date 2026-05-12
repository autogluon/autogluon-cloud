"""Tests for the top-level Python API (autogluon.cloud.initialize/status/teardown)."""

import pytest

import autogluon.cloud as agc
from autogluon.cloud.config import CONFIG_DIR_ENV, load_config, save_config


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    yield tmp_path


@pytest.fixture(autouse=True)
def stub_aws(monkeypatch):
    """Stub out the credential-verification call so tests don't hit AWS."""

    class _StubSession:
        def __init__(self, region):
            self.region_name = region or "us-east-1"

    monkeypatch.setattr(
        "autogluon.cloud.cloud_setup._verified_session",
        lambda aws_profile=None, region=None: _StubSession(region),
    )


def _initialize_existing(**overrides):
    kwargs = dict(
        backend="sagemaker",
        region="us-east-1",
        role_arn="arn:aws:iam::111122223333:role/x",
        bucket="b1",
    )
    kwargs.update(overrides)
    agc.initialize(**kwargs)


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


def test_initialize_with_existing_resources_writes_config():
    """`initialize(role_arn=..., bucket=...)` writes config without CloudFormation."""
    _initialize_existing(region="us-west-2", bucket="my-bucket")

    profile = load_config().get_profile()
    assert profile.region == "us-west-2"
    assert profile.bucket == "my-bucket"
    assert profile.role_arn == "arn:aws:iam::111122223333:role/x"
    assert profile.stack_name is None


def test_initialize_returns_none(capsys):
    """initialize() is a side-effecting setup function; it shouldn't return data."""
    result = _initialize_existing()
    assert result is None
    # And it should print where the config was saved.
    assert "Saved profile" in capsys.readouterr().out


def test_initialize_requires_role_and_bucket_together():
    with pytest.raises(ValueError, match="must be provided together"):
        agc.initialize(role_arn="arn:aws:iam::111122223333:role/x")  # missing bucket


def test_initialize_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        agc.initialize(backend="not-a-backend", role_arn="arn:...", bucket="b")


def test_initialize_overwrites_existing_profile():
    """Second initialize() overwrites the first — no confirmation in Python API."""
    _initialize_existing(bucket="b1")
    _initialize_existing(bucket="b2")
    assert load_config().get_profile().bucket == "b2"


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_without_config_returns_not_found():
    result = agc.status()
    assert result["found"] is False
    assert "reason" not in result  # missing config, not missing profile


def test_status_with_config(monkeypatch):
    """Status returns a snapshot; health checks are stubbed to avoid AWS."""
    _initialize_existing()
    monkeypatch.setattr("autogluon.cloud.cloud_setup._check_bucket", lambda s, b: "ok")
    monkeypatch.setattr("autogluon.cloud.cloud_setup._check_role", lambda s, r: "ok")

    result = agc.status()
    assert result["found"] is True
    assert result["profile_name"] == "default"
    assert result["is_active"] is True
    assert result["checks"] == {"bucket": "ok", "role": "ok"}


def test_status_missing_profile_returns_structured_result():
    """Regression: status(profile_name=<nonexistent>) must not raise KeyError."""
    _initialize_existing()
    result = agc.status(profile_name="nonexistent")
    assert result["found"] is False
    assert result["reason"] == "profile not found"
    assert result["requested_profile"] == "nonexistent"
    assert result["available_profiles"] == ["default"]


def test_status_check_role_false_skips_iam(monkeypatch):
    _initialize_existing()
    monkeypatch.setattr("autogluon.cloud.cloud_setup._check_bucket", lambda s, b: "ok")

    result = agc.status(check_role=False)
    assert "role" not in result["checks"]


# ---------------------------------------------------------------------------
# teardown
# ---------------------------------------------------------------------------


def test_teardown_without_config_is_noop(capsys):
    result = agc.teardown()
    assert result is None
    assert "nothing to tear down" in capsys.readouterr().out


def test_teardown_removes_profile_when_no_stack(capsys):
    """If the profile has no stack (initialize with role_arn/bucket), teardown is safe."""
    _initialize_existing()
    result = agc.teardown()
    assert result is None
    assert load_config().profiles == {}
    assert "no stack to delete" in capsys.readouterr().out


def test_teardown_missing_profile_is_friendly(capsys):
    """Regression: teardown(profile_name=<nonexistent>) must not raise KeyError."""
    _initialize_existing()
    result = agc.teardown(profile_name="nonexistent")
    assert result is None
    assert "not found" in capsys.readouterr().out


def test_teardown_with_stack_hits_cfn(monkeypatch):
    """When a profile has a stack_name, teardown calls cfn.delete_stack."""
    _initialize_existing()

    # Promote the profile to having a fake stack.
    config = load_config()
    config.get_profile().stack_name = "ag-cloud-sagemaker"
    save_config(config)

    calls = []

    class FakeWaiter:
        def wait(self, **kwargs):
            pass

    class FakeCFN:
        def delete_stack(self, StackName):
            calls.append(StackName)

        def get_waiter(self, name):
            assert name == "stack_delete_complete"
            return FakeWaiter()

    class FakeSession:
        def client(self, service):
            assert service == "cloudformation"
            return FakeCFN()

        def resource(self, service):
            raise AssertionError("delete_bucket_contents is False; resource() not expected")

    monkeypatch.setattr(
        "autogluon.cloud.cloud_setup._boto_session",
        lambda aws_profile=None, region=None: FakeSession(),
    )

    agc.teardown()
    assert calls == ["ag-cloud-sagemaker"]
    assert load_config().profiles == {}
