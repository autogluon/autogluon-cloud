import json
import tempfile

import pytest

from autogluon.cloud import TabularCloudPredictor
from autogluon.cloud.backend.constant import RAY_AWS, SAGEMAKER


@pytest.mark.parametrize("backend", [RAY_AWS, SAGEMAKER])
def test_generate_default_permission(backend):
    with tempfile.TemporaryDirectory() as root:
        paths = TabularCloudPredictor.generate_default_permission(
            backend=backend, account_id="foo", cloud_output_bucket="foo", output_path=root
        )
        trust_relationship_path, iam_policy_path = paths["trust_relationship"], paths["iam_policy"]
        for path in [trust_relationship_path, iam_policy_path]:
            with open(path, "r") as file:
                document = json.load(file)
                statement = document.get("Statement", None)
                assert statement is not None
                assert len(statement) > 0
