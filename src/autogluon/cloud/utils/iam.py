import json
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError

from .constants import POLICY_ACCOUNT_PLACE_HOLDER, POLICY_BUCKET_PLACE_HOLDER, TRUST_RELATIONSHIP_ACCOUNT_PLACE_HOLDER


def replace_trust_relationship_place_holder(trust_relationship_document, account_id):
    """Replace placeholder inside template with given values"""
    statements = trust_relationship_document.get("Statement", [])
    for statement in statements:
        for principal in statement["Principal"].keys():
            statement["Principal"][principal] = statement["Principal"][principal].replace(
                TRUST_RELATIONSHIP_ACCOUNT_PLACE_HOLDER, account_id
            )
    return trust_relationship_document


def replace_iam_policy_place_holder(policy_document, account_id=None, bucket=None):
    """Replace placeholder inside template with given values"""
    statements = policy_document.get("Statement", [])
    for statement in statements:
        resources = statement.get("Resource", None)
        if resources is not None:
            if account_id is not None:
                statement["Resource"] = [
                    resource.replace(POLICY_ACCOUNT_PLACE_HOLDER, account_id) for resource in statement["Resource"]
                ]
            if bucket is not None:
                statement["Resource"] = [
                    resource.replace(POLICY_BUCKET_PLACE_HOLDER, bucket) for resource in statement["Resource"]
                ]
    return policy_document


def create_iam_role(role_name: str, trust_relationship: Dict[str, Any]) -> str:
    iam_client = boto3.client("iam")
    try:
        response = iam_client.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(trust_relationship))
        return response["Role"]["Arn"]
    except ClientError as error:
        if error.response["Error"]["Code"] != "EntityAlreadyExists":
            raise error


def create_iam_policy(policy_name: str, policy: Dict[str, Any]) -> str:
    iam_client = boto3.client("iam")
    try:
        response = iam_client.create_policy(PolicyName=policy_name, PolicyDocument=json.dumps(policy))
        return response["Policy"]["Arn"]
    except ClientError as error:
        if error.response["Error"]["Code"] != "EntityAlreadyExists":
            raise error


def attach_iam_policy(role_name: str, policy_arn: str):
    iam_client = boto3.client("iam")
    iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)


def create_instance_profile(instance_profile_name: str) -> str:
    iam_client = boto3.client("iam")
    try:
        response = iam_client.create_instance_profile(InstanceProfileName=instance_profile_name)
        return response["InstanceProfile"]["Arn"]
    except ClientError as error:
        if error.response["Error"]["Code"] != "EntityAlreadyExists":
            raise error


def add_role_to_instance_profile(instance_profile_name: str, role_name: str):
    iam_client = boto3.client("iam")
    iam_client.add_role_to_instance_profile(InstanceProfileName=instance_profile_name, RoleName=role_name)
