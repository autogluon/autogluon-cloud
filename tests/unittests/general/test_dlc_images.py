import subprocess

import pytest

from autogluon.cloud.utils.dlc_utils import retrieve_image_uri


@pytest.mark.parametrize("ag_version", ["1.0.0", "1.1.0", "1.1.1", "1.2.0", "1.3.0", "1.4.0", "1.5.0"])
@pytest.mark.parametrize("instance_type", ["ml.m5.xlarge", "ml.g4dn.xlarge"])
@pytest.mark.parametrize("scope", ["training", "inference"])
def test_dlc_image_exists(ag_version, instance_type, scope):
    uri = retrieve_image_uri(ag_version, "us-east-1", scope, instance_type)
    result = subprocess.run(["docker", "manifest", "inspect", uri], capture_output=True, text=True)
    assert result.returncode == 0, f"Image not found in ECR: {uri}\n{result.stderr}"
