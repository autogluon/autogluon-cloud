import logging
from typing import Optional

import boto3
import sagemaker
from botocore.config import Config

from ..config import load_config

logger = logging.getLogger(__name__)


def resolve_execution_role(role: Optional[str], backend_name: str) -> str:
    """Resolve the SageMaker execution role ARN — the role that *jobs* run as on AWS.

    This is distinct from the *caller* identity (the AWS principal calling
    ``sagemaker:Create*``); the caller never assumes this role. SageMaker assumes it
    on the job's behalf to access S3 inputs/outputs, ECR images, etc.

    Resolution order:

    1. ``role`` argument if provided.
    2. ``role_arn`` from ``~/.autogluon/cloud.yaml`` under the matching backend slot.
    3. ``sagemaker.get_execution_role()`` — only succeeds inside a SageMaker
       Notebook / Studio environment.

    Parameters
    ----------
    role
        Explicit execution role ARN. If given, returned as-is.
    backend_name
        Backend key used to look up the persisted config (e.g. ``"sagemaker"``).

    Returns
    -------
    str
        The resolved execution role ARN.
    """
    if role:
        return role
    config = load_config()
    if config is not None:
        entry = config.backends.get(backend_name)
        if entry is not None and entry.role_arn:
            logger.log(20, f"Using execution role from ~/.autogluon/cloud.yaml: {entry.role_arn}")
            return entry.role_arn
    return sagemaker.get_execution_role()


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
