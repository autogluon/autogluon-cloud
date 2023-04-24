import os
from functools import partial
from typing import Any, Dict, Optional

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
