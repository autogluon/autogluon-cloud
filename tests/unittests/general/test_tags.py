"""Unit tests for the tag-merging helper used by SagemakerBackend."""

import pytest

from autogluon.cloud.utils.tag_utils import DISABLE_DEFAULT_TAGS_ENV, build_tags


def test_when_no_extras_or_user_then_only_module_tag_is_returned():
    assert build_tags("timeseries") == [{"Key": "autogluon-cloud-module", "Value": "timeseries"}]


def test_when_extra_tags_provided_then_appended_after_module():
    tags = build_tags("timeseries", extra_tags=[{"Key": "autogluon-cloud-model-id", "Value": "chronos-2"}])
    assert tags == [
        {"Key": "autogluon-cloud-module", "Value": "timeseries"},
        {"Key": "autogluon-cloud-model-id", "Value": "chronos-2"},
    ]


def test_when_user_tag_collides_with_default_then_user_wins():
    tags = build_tags("timeseries", user_tags=[{"Key": "autogluon-cloud-module", "Value": "override"}])
    assert tags == [{"Key": "autogluon-cloud-module", "Value": "override"}]


def test_when_user_tags_unique_then_appended_after_defaults():
    tags = build_tags("tabular", user_tags=[{"Key": "Owner", "Value": "team"}])
    assert tags == [
        {"Key": "autogluon-cloud-module", "Value": "tabular"},
        {"Key": "Owner", "Value": "team"},
    ]


@pytest.mark.parametrize("value", ["1", "true", "True", "yes"])
def test_when_disable_env_var_set_then_defaults_and_extras_are_skipped(monkeypatch, value):
    """Extras are AG-cloud defaults too — opt-out drops them along with module."""
    monkeypatch.setenv(DISABLE_DEFAULT_TAGS_ENV, value)
    assert build_tags("timeseries") == []
    assert build_tags("timeseries", extra_tags=[{"Key": "autogluon-cloud-model-id", "Value": "chronos-2"}]) == []
    assert build_tags("timeseries", user_tags=[{"Key": "Owner", "Value": "team"}]) == [
        {"Key": "Owner", "Value": "team"}
    ]
