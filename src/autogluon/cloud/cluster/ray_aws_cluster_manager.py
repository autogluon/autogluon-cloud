import json
import logging
import os
import time
from typing import Dict

import boto3

from ..utils.iam import (
    add_role_to_instance_profile,
    attach_iam_policy,
    create_iam_policy,
    create_iam_role,
    create_instance_profile,
    delete_iam_policy,
    get_policy,
    replace_iam_policy_place_holder,
    replace_trust_relationship_place_holder,
)
from ..utils.ray_aws_iam import (
    ECR_READ_ONLY,
    RAY_AWS_CLOUD_POLICY,
    RAY_AWS_IAM_POLICY_FILE_NAME,
    RAY_AWS_POLICY_NAME,
    RAY_AWS_ROLE_NAME,
    RAY_AWS_TRUST_RELATIONSHIP,
    RAY_AWS_TRUST_RELATIONSHIP_FILE_NAME,
    RAY_INSTANCE_PROFILE_NAME,
)
from .ray_cluster_manager import RayClusterManager

logger = logging.getLogger(__name__)


class RayAWSClusterManager(RayClusterManager):
    def __init__(self, config: str, cloud_output_bucket: str, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.cloud_output_bucket = cloud_output_bucket

    @staticmethod
    def generate_default_permission(account_id: str, cloud_output_bucket: str, output_path: str) -> Dict[str, str]:
        """
        Generate trust relationship and iam policy required to manage cluster
        Users can use the generated files to create an IAM role for themselves.
        IMPORTANT: Make sure you review both files before creating the role!

        Parameters
        ----------
        account_id: str
            The AWS account ID you plan to use for the cluster.
        cloud_output_bucket: str
            s3 bucket name where intermediate artifacts will be uploaded and trained models should be saved.
            You need to create this bucket beforehand and we would put this bucket in the policy being created.
        output_path: str
            Where you would like the generated file being written to.
            If not specified, will write to the current folder.

        Return
        ------
        A dict containing the trust relationship and IAM policy files paths
        """
        if output_path is None:
            output_path = "."
        trust_relationship_file_path = os.path.join(output_path, RAY_AWS_TRUST_RELATIONSHIP_FILE_NAME)
        iam_policy_file_path = os.path.join(output_path, RAY_AWS_IAM_POLICY_FILE_NAME)

        trust_relationship = replace_trust_relationship_place_holder(
            trust_relationship_document=RAY_AWS_TRUST_RELATIONSHIP, account_id=account_id
        )
        iam_policy = replace_iam_policy_place_holder(
            policy_document=RAY_AWS_CLOUD_POLICY, account_id=account_id, bucket=cloud_output_bucket
        )
        with open(trust_relationship_file_path, "w") as file:
            json.dump(trust_relationship, file, indent=4)

        with open(iam_policy_file_path, "w") as file:
            json.dump(iam_policy, file, indent=4)

        logger.info(f"Generated trust relationship to {trust_relationship_file_path}")
        logger.info(f"Generated iam policy to {iam_policy_file_path}")
        logger.info(
            "IMPORTANT: Please review the trust relationship and iam policy before you use them to create an IAM role"
        )
        logger.info(
            "Please refer to https://github.com/ray-project/ray/issues/9327 on how to furthur scope down the permission for your use case."
        )
        logger.info(
            "Please refer to AWS documentation on how to create an IAM role: https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create.html"
        )

        return {"trust_relationship": trust_relationship_file_path, "iam_policy": iam_policy_file_path}
