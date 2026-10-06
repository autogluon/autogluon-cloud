import json
import logging
import re
from pathlib import Path

from packaging import version
from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "autogluon_dlc.json"
_GPU_INSTANCE_PREFIXES = ("ml.p", "ml.g")
_INFERENCE_AMI_INSTANCE_PREFIXES = (
    "ml.g4dn.",
    "ml.g5.",
    "ml.g6.",
    "ml.g6e.",
    "ml.p4d.",
    "ml.p4de.",
    "ml.p5.",
    "ml.p5e.",
    "ml.p5en.",
)
_BATCH_AMI_INSTANCE_PREFIXES = ("ml.g4dn.", "ml.g5.", "ml.g6.", "ml.g6e.")
_CUDA_13_AMI_VERSIONS = {
    "inference": "al2023-ami-sagemaker-inference-gpu-4-1",
    "transform": "al2-ami-sagemaker-batch-gpu-535",
}


def _load_config():
    with open(_CONFIG_PATH) as f:
        return json.load(f)


def _is_gpu_instance(instance_type):
    return instance_type.startswith(_GPU_INSTANCE_PREFIXES) or instance_type == "local_gpu"


def infer_sagemaker_ami_version(image_uri, instance_type, image_scope):
    """Infer the SageMaker host AMI required by a CUDA 13 GPU image."""
    assert image_scope in _CUDA_13_AMI_VERSIONS
    instance_prefixes = (
        _INFERENCE_AMI_INSTANCE_PREFIXES if image_scope == "inference" else _BATCH_AMI_INSTANCE_PREFIXES
    )
    if not image_uri or not re.search(r"(?:^|-)cu13\d*(?:-|$)", image_uri.rsplit(":", 1)[-1]):
        return None
    if instance_type.startswith(instance_prefixes):
        return _CUDA_13_AMI_VERSIONS[image_scope]
    return None


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


def retrieve_image_uri(framework_version, region, image_scope, instance_type, py_version=None, custom_image_uri=None):
    """Construct the full ECR image URI for a given AG version/region/scope.

    Drop-in replacement for sagemaker.image_uris.retrieve("autogluon", ...). Returns ``custom_image_uri`` as-is if set.
    """
    if custom_image_uri:
        return custom_image_uri
    config = _load_config()
    version_info = config[image_scope]["versions"][framework_version]
    registry = version_info["registries"][region]
    repository = version_info["repository"]
    processor = "gpu" if _is_gpu_instance(instance_type) else "cpu"
    if py_version is None:
        py_version = version_info["py_versions"][0]
    os_suffix = version_info.get("os")
    cuda_version = version_info.get("cuda_version")
    if version_info.get("unified"):
        # AG 1.6+ ships a single image for training and inference, tagged e.g. 1.6.3-cpu-amzn2023 / 1.6.3-cu133-amzn2023
        accelerator = cuda_version if processor == "gpu" else "cpu"
        tag = f"{framework_version}-{accelerator}-{os_suffix}"
    elif os_suffix:
        if processor == "gpu" and cuda_version:
            tag = f"{framework_version}-{processor}-{py_version}-{cuda_version}-{os_suffix}"
        else:
            tag = f"{framework_version}-{processor}-{py_version}-{os_suffix}"
    else:
        tag = f"{framework_version}-{processor}-{py_version}"
    return f"{registry}.dkr.ecr.{region}.amazonaws.com/{repository}:{tag}"


def resolve_framework_version(framework_version, framework_type="training"):
    """Resolve ``framework_version`` to a version that has an official AutoGluon DLC image.

    Accepts ``"latest"``, ``"x.y"`` or ``"x.y.z"``. ``"x.y"`` resolves to the newest patch release of ``x.y`` with an
    image. ``"x.y.z"`` without an image falls back to the newest patch release of ``x.y``, since patch releases of
    AutoGluon are compatible with each other.
    """
    available = sorted(retrieve_available_framework_versions(framework_type), key=version.parse)
    if framework_version == "latest":
        return available[-1]
    if framework_version in available:
        return framework_version
    supported_minors = sorted({".".join(v.split(".")[:2]) for v in available}, key=version.parse)
    try:
        requested = Version(framework_version)
    except InvalidVersion:
        raise ValueError(
            f"Invalid framework_version={framework_version!r}. Use 'latest' or one of {supported_minors}."
        ) from None
    same_minor = [
        v for v in available if len(requested.release) >= 2 and Version(v).release[:2] == requested.release[:2]
    ]
    if not same_minor:
        raise ValueError(
            f"No AutoGluon container available for framework_version={framework_version!r}. "
            f"Supported versions: {supported_minors}."
        )
    resolved = same_minor[-1]
    if len(requested.release) > 2:
        logger.warning(f"No AutoGluon container available for version {framework_version}, using {resolved} instead.")
    return resolved


def infer_framework_version_from_image_uri(image_uri):
    """Return the AutoGluon version of an official DLC training image URI, or None for any other image."""
    repository, _, tag = image_uri.rsplit("/", 1)[-1].partition(":")
    framework_version = tag.split("-", 1)[0]
    version_info = _load_config()["training"]["versions"].get(framework_version)
    if version_info is None or version_info["repository"] != repository:
        return None
    return framework_version


def parse_framework_version(framework_version, framework_type, py_version=None, minimum_version=None):
    framework_version = resolve_framework_version(framework_version, framework_type)
    if minimum_version is not None and Version(framework_version) < Version(minimum_version):
        raise ValueError(f"Cloud module only supports {minimum_version}+ containers.")
    valid_py_versions = retrieve_py_versions(framework_version, framework_type)
    if py_version is not None:
        assert py_version in valid_py_versions, f"{py_version} is no a valid option. Options are {valid_py_versions}"
    else:
        py_version = valid_py_versions[0]
    return framework_version, py_version
