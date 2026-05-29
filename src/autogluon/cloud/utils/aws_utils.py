import logging
from typing import Optional

import boto3
import sagemaker
from botocore.config import Config

from autogluon.common.utils.s3_utils import is_s3_url

from ..config import load_config

logger = logging.getLogger(__name__)


def resolve_execution_role(role: Optional[str], backend_name: str) -> str:
    """Resolve the SageMaker execution role ARN.

    Resolution order:

    1. ``role`` argument if provided.
    2. ``role_arn`` from ``~/.autogluon/cloud.yaml`` under the matching backend slot.
    3. ``sagemaker.get_execution_role()``.
    """
    if role:
        return role
    config = load_config()
    if config is not None:
        entry = config.backends.get(backend_name)
        if entry is not None and entry.role_arn:
            logger.info(f"Using execution role from ~/.autogluon/cloud.yaml: {entry.role_arn}")
            return entry.role_arn
    return sagemaker.get_execution_role()


def resolve_cloud_output_path(path: Optional[str], backend_name: str) -> str:
    """Resolve the S3 location where AutoGluon-Cloud will read/write artifacts.

    Resolution order for the bucket:

    1. ``path`` argument if provided (``s3://bucket`` or ``s3://bucket/prefix``).
    2. ``bucket`` from ``~/.autogluon/cloud.yaml`` under the matching backend slot.

    Prefix behavior:

    * Bucket only (no prefix) — a unique timestamped subfolder ``ag-<timestamp>`` is appended.
      Each call gets its own folder, so repeated runs don't overwrite each other.
    * Bucket and prefix — the path is used verbatim. Re-running with the same prefix
      will overwrite previously written artifacts; pick a fresh prefix per run if you
      want them kept side by side.

    Raises ``ValueError`` if no path is given and no bucket is configured.
    """
    if path is None:
        config = load_config()
        entry = config.backends.get(backend_name) if config is not None else None
        if entry is None or not entry.bucket:
            raise ValueError(
                "No `cloud_output_path` was provided and no bucket is configured for backend "
                f"{backend_name!r} in ~/.autogluon/cloud.yaml. Either pass `cloud_output_path=` "
                "explicitly, or run `autogluon.cloud.bootstrap()` / `register(bucket=...)` once "
                "to persist a bucket."
            )
        path = f"s3://{entry.bucket}"
        logger.info(f"Using bucket from ~/.autogluon/cloud.yaml: {entry.bucket}")

    path = path.rstrip("/")
    if not is_s3_url(path):
        path = "s3://" + path
    body = path[len("s3://") :]
    bucket, _, prefix = body.partition("/")
    if not prefix:
        path = f"s3://{bucket}/ag-{sagemaker.utils.sagemaker_timestamp()}"
        logger.info(f"cloud_output_path set to {path} (timestamped subfolder under bucket).")
    else:
        logger.info(f"cloud_output_path set to {path}.")
        if _s3_prefix_has_objects(bucket, prefix):
            logger.warning(
                f"cloud_output_path {path} already contains objects. Running fit()/deploy() "
                "will overwrite the existing artifacts. Pass a fresh prefix, or pass just "
                "`s3://<bucket>` to get a unique timestamped subfolder."
            )
    return path


def _s3_prefix_has_objects(bucket: str, prefix: str) -> bool:
    """Return True if any object exists under ``s3://bucket/prefix``.

    Swallows errors (missing credentials, AccessDenied, NoSuchBucket) and returns False —
    we use this only for an advisory warning, so it must never break construction.
    """
    try:
        response = boto3.client("s3").list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        return response.get("KeyCount", 0) > 0
    except Exception as e:
        logger.debug(f"Skipping cloud_output_path emptiness check ({type(e).__name__}: {e})")
        return False


def get_latest_amazon_linux_ami(region="us-east-1", version="al2023"):
    ec2_client = boto3.client("ec2", region_name=region)
    filters = [
        {"Name": "owner-alias", "Values": ["amazon"]},
        {"Name": "name", "Values": [f"{version}-ami-*"]},  # Amazon Linux 2 or AL2023
        {"Name": "state", "Values": ["available"]},
    ]
    response = ec2_client.describe_images(Filters=filters, Owners=["amazon"])
    if not response["Images"]:
        raise ValueError("No Amazon Linux AMI found!")

    latest_ami = sorted(response["Images"], key=lambda x: x["CreationDate"], reverse=True)[0]
    return latest_ami["ImageId"]


def setup_sagemaker_session(
    config: Optional[Config] = None,
    connect_timeout: int = 60,
    read_timeout: int = 60,
    retries: Optional[dict] = None,
    **kwargs,
):
    """
    Setup a sagemaker session with a given configuration

    Parameters
    ----------
    config
        A botocore.Config object providing the intended configuration
        https://botocore.amazonaws.com/v1/documentation/api/latest/reference/config.html
    connect_timeout
        The time in seconds till a timeout exception is thrown when attempting to make a connection.
        The default is 60 seconds.
    read_timeout
        The time in seconds till a timeout exception is thrown when attempting to read from a connection.
        The default is 60 seconds.
    retries
        A dictionary for retry specific configurations. Valid keys are:
            'total_max_attempts' -- An integer representing the maximum number of total attempts that will be made on a single request.
                This includes the initial request, so a value of 1 indicates that no requests will be retried.
                If total_max_attempts and max_attempts are both provided, total_max_attempts takes precedence.
                total_max_attempts is preferred over max_attempts because it maps to the AWS_MAX_ATTEMPTS environment variable
                and the max_attempts config file value.
            'max_attempts' -- An integer representing the maximum number of retry attempts that will be made on a single request.
                For example, setting this value to 2 will result in the request being retried at most two times after the initial request.
                Setting this value to 0 will result in no retries ever being attempted on the initial request.
                If not provided, the number of retries will default to whatever is modeled, which is typically four retries.
            'mode' -- A string representing the type of retry mode botocore should use. Valid values are:
                legacy - The pre-existing retry behavior.
                standard - The standardized set of retry rules. This will also default to 3 max attempts unless overridden.
                adaptive - Retries with additional client side throttling.
    """
    if config is None:
        if retries is None:
            retries = {"max_attempts": 20}
        config = Config(connect_timeout=connect_timeout, read_timeout=read_timeout, retries=retries, **kwargs)
    sm_boto = boto3.client("sagemaker", config=config)
    return sagemaker.Session(sagemaker_client=sm_boto)
