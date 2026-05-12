"""Tests for the top-level Python API (autogluon.cloud.init/status/teardown)."""

import pytest

import autogluon.cloud as agc
from autogluon.cloud.config import CONFIG_DIR_ENV, load_config


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    yield tmp_path


@pytest.fixture(autouse=True)
def stub_detect_identity(monkeypatch):
    """Don't hit real AWS in unit tests."""
    monkeypatch.setattr(
        "autogluon.cloud._setup.detect_aws_identity",
        lambda region=None, aws_profile=None: {
            "account": "111122223333",
            "arn": "arn:aws:iam::111122223333:user/test",
            "region": region or "us-east-1",
        },
    )


def test_init_with_existing_resources_non_interactive():
    """`init(role_arn=..., bucket=..., yes=True)` writes config without CFN."""
    result = agc.init(
        backend="sagemaker",
        region="us-west-2",
        role_arn="arn:aws:iam::111122223333:role/ag-cloud",
        bucket="my-existing-bucket",
        yes=True,
        interactive=False,
    )
    assert result["profile_name"] == "default"
    assert result["bucket"] == "my-existing-bucket"
    assert result["role_arn"] == "arn:aws:iam::111122223333:role/ag-cloud"
    assert result["stack_name"] is None

    config = load_config()
    assert config is not None
    profile = config.get_profile()
    assert profile.region == "us-west-2"
    assert profile.bucket == "my-existing-bucket"


def test_init_requires_role_and_bucket_together():
    with pytest.raises(ValueError, match="must be provided together"):
        agc.init(
            backend="sagemaker",
            region="us-east-1",
            role_arn="arn:aws:iam::111122223333:role/ag-cloud",
            # missing bucket
            yes=True,
            interactive=False,
        )


def test_init_refuses_to_overwrite_without_yes():
    agc.init(
        backend="sagemaker",
        region="us-east-1",
        role_arn="arn:aws:iam::111122223333:role/x",
        bucket="b1",
        yes=True,
        interactive=False,
    )
    with pytest.raises(RuntimeError, match="already exists"):
        agc.init(
            backend="sagemaker",
            region="us-east-1",
            role_arn="arn:aws:iam::111122223333:role/y",
            bucket="b2",
            yes=False,
            interactive=False,
        )


def test_init_refuses_to_deploy_without_yes_in_non_interactive():
    """Safety: non-interactive + no yes + no existing resources = refuse."""
    with pytest.raises(RuntimeError, match="Refusing to deploy"):
        agc.init(
            backend="sagemaker",
            region="us-east-1",
            yes=False,
            interactive=False,
        )


def test_status_without_config(capsys):
    result = agc.status()
    assert result["found"] is False
    # Prints a hint pointing users at the Python init function.
    out = capsys.readouterr().out
    assert "autogluon.cloud.init()" in out


def test_status_with_config(monkeypatch):
    """Status returns a snapshot; health checks are stubbed to avoid AWS calls."""
    agc.init(
        backend="sagemaker",
        region="us-east-1",
        role_arn="arn:aws:iam::111122223333:role/x",
        bucket="b1",
        yes=True,
        interactive=False,
    )
    # Stub the health checks so we don't reach AWS.
    monkeypatch.setattr("autogluon.cloud._setup._check_bucket", lambda s, b: "ok")
    monkeypatch.setattr("autogluon.cloud._setup._check_role", lambda p: "ok")

    result = agc.status()
    assert result["found"] is True
    assert result["profile_name"] == "default"
    assert result["is_active"] is True
    assert result["checks"]["bucket"] == "ok"
    assert result["checks"]["role"] == "ok"


def test_teardown_without_config():
    result = agc.teardown(yes=True)
    assert result["removed"] is False
    assert result["reason"] == "no config"


def test_teardown_refuses_without_yes_in_non_interactive():
    agc.init(
        backend="sagemaker",
        region="us-east-1",
        role_arn="arn:aws:iam::111122223333:role/x",
        bucket="b1",
        yes=True,
        interactive=False,
    )
    # Set a stack_name on the profile to exercise the destructive path.
    config = load_config()
    profile = config.get_profile()
    profile.stack_name = "ag-cloud-sagemaker"
    from autogluon.cloud.config import save_config

    save_config(config)

    with pytest.raises(RuntimeError, match="Refusing to tear down"):
        agc.teardown(yes=False, interactive=False)


def test_teardown_removes_profile_when_no_stack():
    """If the profile has no stack (init with role_arn/bucket), teardown is safe."""
    agc.init(
        backend="sagemaker",
        region="us-east-1",
        role_arn="arn:aws:iam::111122223333:role/x",
        bucket="b1",
        yes=True,
        interactive=False,
    )
    result = agc.teardown(yes=True, interactive=False)
    assert result["removed"] is True
    assert result["stack_deleted"] is False
    assert load_config().profiles == {}


def test_teardown_missing_profile_returns_structured_result():
    """Regression: teardown(profile_name=<nonexistent>) must not raise KeyError."""
    agc.init(
        backend="sagemaker",
        region="us-east-1",
        role_arn="arn:aws:iam::111122223333:role/x",
        bucket="b1",
        yes=True,
        interactive=False,
    )
    result = agc.teardown(profile_name="nonexistent", yes=True, interactive=False)
    assert result["removed"] is False
    assert result["reason"] == "profile not found"
    assert result["requested_profile"] == "nonexistent"
    assert result["available_profiles"] == ["default"]


def test_status_missing_profile_returns_structured_result():
    """Regression: status(profile_name=<nonexistent>) must not raise KeyError."""
    agc.init(
        backend="sagemaker",
        region="us-east-1",
        role_arn="arn:aws:iam::111122223333:role/x",
        bucket="b1",
        yes=True,
        interactive=False,
    )
    result = agc.status(profile_name="nonexistent")
    assert result["found"] is False
    assert result["reason"] == "profile not found"
    assert result["requested_profile"] == "nonexistent"
    assert result["available_profiles"] == ["default"]
