import boto3
import pytest

from autogluon.cloud.utils.dlc_utils import retrieve_image_uri


@pytest.mark.parametrize("ag_version", ["1.0.0", "1.1.0", "1.1.1", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.3"])
@pytest.mark.parametrize("instance_type", ["ml.m5.xlarge", "ml.g4dn.xlarge"])
@pytest.mark.parametrize("scope", ["training", "inference"])
def test_dlc_image_exists(ag_version, instance_type, scope):
    region = "us-east-1"
    uri = retrieve_image_uri(ag_version, region, scope, instance_type)
    repository, tag = uri.split("/", 1)[1].split(":")
    registry_id = uri.split(".")[0]
    ecr_client = boto3.client("ecr", region_name=region)
    response = ecr_client.describe_images(
        registryId=registry_id,
        repositoryName=repository,
        imageIds=[{"imageTag": tag}],
    )
    assert len(response["imageDetails"]) == 1, f"Image not found in ECR: {uri}"


@pytest.mark.parametrize(
    ("instance_type", "expected_tag"),
    [("ml.m5.xlarge", "1.6.3-cpu-amzn2023"), ("ml.g4dn.xlarge", "1.6.3-cu133-amzn2023")],
)
@pytest.mark.parametrize("scope", ["training", "inference"])
def test_unified_dlc_image_uri(instance_type, expected_tag, scope):
    uri = retrieve_image_uri("1.6.3", "us-east-1", scope, instance_type)
    assert uri == f"763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon:{expected_tag}"


@pytest.mark.parametrize("backend", ["ray", "ray_aws"])
def test_ray_backend_raises_removed_error(backend, tmp_path):
    from autogluon.cloud import TabularCloudPredictor

    with pytest.raises(ValueError, match="removed in AutoGluon-Cloud v0.7.0"):
        TabularCloudPredictor(backend=backend, local_output_path=str(tmp_path))
