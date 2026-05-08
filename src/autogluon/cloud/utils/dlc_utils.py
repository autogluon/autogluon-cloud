import json
from pathlib import Path

from packaging import version
from packaging.version import Version

_CONFIG_PATH = Path(__file__).parent / "autogluon_dlc.json"
_GPU_INSTANCE_PREFIXES = ("ml.p", "ml.g")


def _load_config():
    with open(_CONFIG_PATH) as f:
        return json.load(f)


def _is_gpu_instance(instance_type):
    return instance_type.startswith(_GPU_INSTANCE_PREFIXES)


def retrieve_available_framework_versions(framework_type="training", details=False):
    """Get available versions of autogluon

    Args:
        framework_type (str, optional):
            Type of framework. Options: 'training', 'inference'.
            Defaults to 'training'.
        details (bool, optional):
            Whether to get detailed information of each versions.
            Defaults to False.

    Returns:
        (Union(list, dict)):
            returns a list of versions if detailed == False.
            returns a dict containing information related to each version if detailed == True.
    """
    assert framework_type in ["training", "inference"]
    config = _load_config()
    versions_details = config[framework_type]["versions"]
    if details:
        return versions_details
    return list(versions_details.keys())


def retrieve_py_versions(framework_version, framework_type="training"):
    versions_details = retrieve_available_framework_versions(framework_type, details=True)
    return versions_details[framework_version]["py_versions"]


def retrieve_latest_framework_version(framework_type="training"):
    """Get latest version of autogluon framework and its py_versions

    Args:
        framework_type (str, optional):
            Type of framework. Options: 'training', 'inference'.
            Defaults to 'training'.

    Returns:
        (str, list):
            version number of latest autogluon framework, and its py_versions as a list
    """
    versions = retrieve_available_framework_versions(framework_type)
    versions.sort(key=version.parse)
    versions = [(v, retrieve_py_versions(v, framework_type)) for v in versions]
    return versions[-1]


def retrieve_image_uri(framework_version, region, image_scope, instance_type, py_version=None):
    """Construct the full ECR image URI for a given AG version/region/scope.

    Drop-in replacement for sagemaker.image_uris.retrieve("autogluon", ...).
    """
    config = _load_config()
    version_info = config[image_scope]["versions"][framework_version]
    registry = version_info["registries"][region]
    repository = version_info["repository"]
    processor = "gpu" if _is_gpu_instance(instance_type) else "cpu"
    if py_version is None:
        py_version = version_info["py_versions"][0]
    os_suffix = version_info.get("os")
    cuda_version = version_info.get("cuda_version")
    if os_suffix:
        if processor == "gpu" and cuda_version:
            tag = f"{framework_version}-{processor}-{py_version}-{cuda_version}-{os_suffix}"
        else:
            tag = f"{framework_version}-{processor}-{py_version}-{os_suffix}"
    else:
        tag = f"{framework_version}-{processor}-{py_version}"
    return f"{registry}.dkr.ecr.{region}.amazonaws.com/{repository}:{tag}"


def parse_framework_version(framework_version, framework_type, py_version=None, minimum_version=None):
    if framework_version == "latest":
        framework_version, py_versions = retrieve_latest_framework_version(framework_type)
        py_version = py_versions[0]
    else:
        if minimum_version is not None and Version(framework_version) < Version(minimum_version):
            raise ValueError("Cloud module only supports 0.6+ containers.")
        valid_options = retrieve_available_framework_versions(framework_type)
        assert (
            framework_version in valid_options
        ), f"{framework_version} is not a valid option. Options are: {valid_options}"

        valid_py_versions = retrieve_py_versions(framework_version, framework_type)
        if py_version is not None:
            assert (
                py_version in valid_py_versions
            ), f"{py_version} is no a valid option. Options are {valid_py_versions}"
        else:
            py_version = valid_py_versions[0]
    return framework_version, py_version
