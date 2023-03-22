from distutils.version import StrictVersion

from packaging import version
from sagemaker import image_uris


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
    config = image_uris.config_for_framework("autogluon")
    versions_details = config[framework_type]["versions"]
    if details:
        return versions_details
    versions = list(config[framework_type]["versions"].keys())
    return versions


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
    versions.sort(key=StrictVersion)
    versions = [(v, retrieve_py_versions(v, framework_type)) for v in versions]
    return versions[-1]


def parse_framework_version(framework_version, framework_type, py_version=None, minimum_version=None):
    if framework_version == "latest":
        framework_version, py_versions = retrieve_latest_framework_version(framework_type)
        py_version = py_versions[0]
    else:
        # Cloud supports 0.6+ containers
        if minimum_version is not None and version.parse(framework_version) < version.parse(minimum_version):
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
