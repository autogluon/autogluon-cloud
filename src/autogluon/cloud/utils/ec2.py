import os
from functools import partial
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError


def _get_key_pair(key_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve info of a key pair. Return None if not found
    To learn more about the info structure: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/describe_key_pairs.html#
    """
    ec2 = boto3.client("ec2")
    try:
        response = ec2.describe_key_pairs(KeyNames=[key_name])
        return response["KeyPairs"][0]
    except ClientError as error:
        if error.response["Error"]["Code"] == "InvalidKeyPair.NotFound":
            return None
        raise error


def create_key_pair(key_name: str, local_path: str) -> str:
    """
    Create a key pair and store the private key to the local path with name `f"{key_name}.pem"`.

    Parameters
    ----------
    key_name: str
        Name of the key pair to create
    local_path: str
        Path to store the private key

    Return
    ------
    str,
        Path to the local private key
    """
    assert _get_key_pair(key_name) is None, f"Key {key_name} already exists. Please choose a different name"
    ec2 = boto3.client("ec2")
    key = ec2.create_key_pair(KeyName=key_name)

    local_path = os.path.join(local_path, f"{key_name}.pem")
    with open(local_path, "w", opener=partial(os.open, mode=0o600)) as f:
        f.write(key["KeyMaterial"])
    return local_path


def delete_key_pair(key_name: str, local_path: Optional[str]):
    """
    Delete a key pair and if local_path is provided, will try to look for matching name private key and delete
    """
    if _get_key_pair(key_name) is None:
        return
    ec2 = boto3.client("ec2")
    ec2.delete_key_pair(KeyName=key_name)
    if local_path is not None:
        local_path = os.path.join(local_path, f"{key_name}.pem")
        if os.path.exists(local_path):
            os.remove(local_path)


def get_latest_ami(ami_name: str = "Deep Learning AMI GPU PyTorch*Ubuntu*") -> str:
    """
    Get the latest ami id

    Parameter
    ---------
    ami_name: str, default = Deep Learning AMI GPU PyTorch*Ubuntu*
        Name of the ami. Could be regex.

    Return
    ------
    str,
        The latest ami id of the ami name being specified
    """
    from dateutil import parser

    def newest_image(list_of_images: List[Dict[str, Any]]):
        latest = None

        for image in list_of_images:
            if not latest:
                latest = image
                continue

            if parser.parse(image["CreationDate"]) > parser.parse(latest["CreationDate"]):
                latest = image

        return latest

    ec2 = boto3.client("ec2")

    filters = [
        {"Name": "name", "Values": [ami_name]},
        {"Name": "owner-alias", "Values": ["amazon"]},
        {"Name": "architecture", "Values": ["x86_64"]},
        {"Name": "state", "Values": ["available"]},
    ]
    response = ec2.describe_images(Owners=["amazon"], Filters=filters)
    source_image = newest_image(response["Images"])
    return source_image["ImageId"]
