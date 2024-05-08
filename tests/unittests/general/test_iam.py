import boto3
from moto import mock_aws

from autogluon.cloud.utils.iam import (
    add_role_to_instance_profile,
    attach_iam_policy,
    create_iam_policy,
    create_iam_role,
    create_instance_profile,
    delete_iam_policy,
    get_policy,
)


@mock_aws
def test_iam_utils():
    iam_client = boto3.client("iam")
    dummy_role = "dummy_role"
    dummy_trust_relationship = {}
    dummy_role_arn = create_iam_role(dummy_role, dummy_trust_relationship)
    assert dummy_role_arn is not None
    create_iam_role(dummy_role, dummy_trust_relationship)  # check for recreation
    dummy_policy = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Action": "none:null", "Resource": "*"}],
    }
    dummy_policy_name = "dommy_policy"
    dummy_policy_arn = create_iam_policy(dummy_policy_name, dummy_policy)
    assert dummy_policy_arn is not None
    attach_iam_policy(dummy_role, dummy_policy_arn)
    attached_policy = iam_client.list_attached_role_policies(RoleName=dummy_role)["AttachedPolicies"]
    assert attached_policy[0]["PolicyArn"] == dummy_policy_arn
    dummy_instance_profile = "dummy_instance_profile"
    create_instance_profile(dummy_instance_profile)
    add_role_to_instance_profile(dummy_instance_profile, dummy_role)
    instance_profile = iam_client.get_instance_profile(InstanceProfileName=dummy_instance_profile)["InstanceProfile"]
    assert instance_profile["Roles"][0]["Arn"] == dummy_role_arn
    policy_arn = get_policy(dummy_policy_name, scope="Local")
    delete_iam_policy(policy_arn)
