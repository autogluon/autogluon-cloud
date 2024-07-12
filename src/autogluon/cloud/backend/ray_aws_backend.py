from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import boto3

from ..cluster.ray_aws_cluster_config_generator import RayAWSClusterConfigGenerator
from ..cluster.ray_aws_cluster_manager import RayAWSClusterManager
from ..cluster.ray_cluster_config_generator import RayClusterConfigGenerator
from ..cluster.ray_cluster_manager import RayClusterManager
from ..utils.ec2 import create_key_pair, delete_key_pair
from ..utils.iam import (
    add_role_to_instance_profile,
    attach_iam_policy,
    create_iam_policy,
    create_iam_role,
    create_instance_profile,
    get_policy,
    replace_iam_policy_place_holder,
    replace_trust_relationship_place_holder,
)
from ..utils.ray_aws_iam import (
    ECR_READ_ONLY,
    RAY_AWS_CLOUD_POLICY,
    RAY_AWS_POLICY_NAME,
    RAY_AWS_ROLE_NAME,
    RAY_AWS_TRUST_RELATIONSHIP,
    RAY_INSTANCE_PROFILE_NAME,
)
from .constant import RAY_AWS, TABULAR_RAY_AWS
from .ray_backend import RayBackend

logger = logging.getLogger(__name__)


class RayAWSBackend(RayBackend):
    name = RAY_AWS

    @property
    def _cluster_config_generator(self) -> RayClusterConfigGenerator:
        return RayAWSClusterConfigGenerator

    @property
    def _cluster_manager(self) -> RayClusterManager:
        return RayAWSClusterManager

    @property
    def _config_file_name(self) -> str:
        return "ag_ray_aws_cluster_config.yaml"

    def initialize(self, **kwargs) -> None:
        """Initialize the backend."""
        super().initialize(**kwargs)
        self._boto_session = boto3.session.Session()
        self.region = self._boto_session.region_name
        assert (
            self.region is not None
        ), "Please setup a region via `export AWS_DEFAULT_REGION=YOUR_REGION` in the terminal"

    @staticmethod
    def generate_default_permission(**kwargs) -> Dict[str, str]:
        """Generate default permission file user could use to setup the corresponding entity, i.e. IAM Role in AWS"""
        return RayAWSClusterManager.generate_default_permission(**kwargs)

    def _setup_role_and_permission(self):
        """
        AutoGluon distributed training requires access to s3 bucket and ecr repo.
        This means the default role being created by ray is not enough.
        """
        account_id = boto3.client("sts").get_caller_identity().get("Account")
        cloud_output_bucket = self.cloud_output_path
        trust_relationship = replace_trust_relationship_place_holder(
            trust_relationship_document=RAY_AWS_TRUST_RELATIONSHIP, account_id=account_id
        )
        iam_policy = replace_iam_policy_place_holder(
            policy_document=RAY_AWS_CLOUD_POLICY, account_id=account_id, bucket=cloud_output_bucket
        )
        create_iam_role(role_name=RAY_AWS_ROLE_NAME, trust_relationship=trust_relationship)
        policy_arn = get_policy(policy_name=RAY_AWS_POLICY_NAME)
        if policy_arn is None:
            policy_arn = create_iam_policy(policy_name=RAY_AWS_POLICY_NAME, policy=iam_policy)
            attach_iam_policy(role_name=RAY_AWS_ROLE_NAME, policy_arn=policy_arn)
            attach_iam_policy(role_name=RAY_AWS_ROLE_NAME, policy_arn=ECR_READ_ONLY)
        instance_profile_arn = create_instance_profile(instance_profile_name=RAY_INSTANCE_PROFILE_NAME)
        if instance_profile_arn is not None:
            add_role_to_instance_profile(instance_profile_name=RAY_INSTANCE_PROFILE_NAME, role_name=RAY_AWS_ROLE_NAME)
        time.sleep(5)  # Leave sometime to allow resource to propagate

    def _setup_key(self, key_name: str, local_path: str) -> str:
        return create_key_pair(key_name=key_name, local_path=local_path)

    def _cleanup_key(self, key_name: str, local_path: Optional[str]):
        delete_key_pair(key_name=key_name, local_path=local_path)


class TabularRayAWSBackend(RayAWSBackend):
    name = TABULAR_RAY_AWS
