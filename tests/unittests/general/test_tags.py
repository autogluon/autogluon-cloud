"""Unit tests for the tag-merging helper used by SagemakerBackend."""

import pytest

from autogluon.cloud.utils.tag_utils import DISABLE_DEFAULT_TAGS_ENV, build_tags, to_request_tags


def test_when_no_extras_or_user_then_only_module_tag_is_returned():
    assert build_tags("timeseries") == {"autogluon-cloud-module": "timeseries"}


def test_when_extra_tags_provided_then_added_to_module_tag():
    tags = build_tags("timeseries", extra_tags={"autogluon-cloud-model-id": "chronos-2"})
    assert tags == {"autogluon-cloud-module": "timeseries", "autogluon-cloud-model-id": "chronos-2"}


def test_when_user_tag_collides_with_default_then_user_wins():
    tags = build_tags("timeseries", user_tags={"autogluon-cloud-module": "override"})
    assert tags == {"autogluon-cloud-module": "override"}


def test_when_user_tags_unique_then_added_to_defaults():
    tags = build_tags("tabular", user_tags={"Owner": "team"})
    assert tags == {"autogluon-cloud-module": "tabular", "Owner": "team"}


@pytest.mark.parametrize("value", ["1", "true", "True", "yes"])
def test_when_disable_env_var_set_then_defaults_and_extras_are_skipped(monkeypatch, value):
    """Extras are AG-cloud defaults too — opt-out drops them along with module."""
    monkeypatch.setenv(DISABLE_DEFAULT_TAGS_ENV, value)
    assert build_tags("timeseries") == {}
    assert build_tags("timeseries", extra_tags={"autogluon-cloud-model-id": "chronos-2"}) == {}
    assert build_tags("timeseries", user_tags={"Owner": "team"}) == {"Owner": "team"}


def test_to_request_tags_uses_api_field_names():
    assert to_request_tags({"Owner": "team"}) == [{"Key": "Owner", "Value": "team"}]
