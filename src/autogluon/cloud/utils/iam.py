import json
from typing import Any, Dict, Optional

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
                if bucket.startswith("s3://"):
                    bucket = bucket[5:]
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


def get_instance_profile_arn(instance_profile_name: str) -> str:
    iam_client = boto3.client("iam")
    response = iam_client.get_instance_profile(InstanceProfileName=instance_profile_name)
    return response["InstanceProfile"]["Arn"]


def delete_iam_policy(policy_arn: str):
    """
    To delete a policy, need to first detach the policy to all attached entities, then delete all versions, finally delete the policy
    """
    detach_policy(policy_arn=policy_arn)
    delete_policy_versions(policy_arn=policy_arn)
    iam_client = boto3.client("iam")
    iam_client.delete_policy(PolicyArn=policy_arn)


def detach_policy(policy_arn: str):
    iam_client = boto3.client("iam")
    response = iam_client.list_entities_for_policy(PolicyArn=policy_arn)
    policy_groups = response.get("PolicyGroups", [])
    policy_users = response.get("PolicyUsers", [])
    policy_roles = response.get("PolicyRoles", [])

    for group in policy_groups:
        group_name = group["GroupName"]
        iam_client.detach_group_policy(GroupName=group_name, PolicyArn=policy_arn)
    for user in policy_users:
        user_name = user["UserName"]
        iam_client.detach_user_policy(UserName=user_name, PolicyArn=policy_arn)
    for role in policy_roles:
        role_name = role["RoleName"]
        iam_client.detach_role_policy(RoleName=role_name, PolicyArn=policy_arn)


def delete_policy_versions(policy_arn: str):
    iam_client = boto3.client("iam")
    response = iam_client.list_policy_versions(PolicyArn=policy_arn)
    version_ids = [version["VersionId"] for version in response["Versions"] if version["IsDefaultVersion"] is False]
    for version_id in version_ids:
        iam_client.delete_policy_version(PolicyArn=policy_arn, VersionId=version_id)


def get_policy(policy_name: str, scope: str = "All") -> Optional[str]:
    iam_client = boto3.client("iam")
    response = iam_client.list_policies(Scope="Local")
    policies = response["Policies"]
    for policy in policies:
        if policy["PolicyName"] == policy_name:
            return policy["Arn"]
    return None
