from .constants import POLICY_ACCOUNT_PLACE_HOLDER, POLICY_BUCKET_PLACE_HOLDER

RAY_AWS_TRUST_RELATIONSHIP_FILE_NAME = "ag_cloud_ray_aws_trust_relationship.json"
RAY_AWS_IAM_POLICY_FILE_NAME = "ag_cloud_ray_aws_iam_policy.json"

RAY_AWS_ROLE_NAME = "AGRayCluster-v1"
RAY_AWS_POLICY_NAME = "AGRayClusterPolicy-v1"
RAY_INSTANCE_PROFILE_NAME = RAY_AWS_ROLE_NAME

RAY_AWS_TRUST_RELATIONSHIP = {
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}],
}

RAY_AWS_CLOUD_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {"Effect": "Allow", "Action": "ec2:RunInstances", "Resource": ["arn:aws:ec2:*::image/ami-*"]},
        {
            "Effect": "Allow",
            "Action": "ec2:RunInstances",
            "Resource": [
                f"arn:aws:ec2:*:{POLICY_ACCOUNT_PLACE_HOLDER}:instance/*",
                f"arn:aws:ec2:*:{POLICY_ACCOUNT_PLACE_HOLDER}:network-interface/*",
                f"arn:aws:ec2:*:{POLICY_ACCOUNT_PLACE_HOLDER}:subnet/*",
                f"arn:aws:ec2:*:{POLICY_ACCOUNT_PLACE_HOLDER}:key-pair/*",
                f"arn:aws:ec2:*:{POLICY_ACCOUNT_PLACE_HOLDER}:volume/*",
                f"arn:aws:ec2:*:{POLICY_ACCOUNT_PLACE_HOLDER}:security-group/*",
            ],
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:TerminateInstances",
                "ec2:DeleteTags",
                "ec2:StartInstances",
                "ec2:CreateTags",
                "ec2:StopInstances",
            ],
            "Resource": [f"arn:aws:ec2:*:{POLICY_ACCOUNT_PLACE_HOLDER}:instance/*"],
        },
        {"Effect": "Allow", "Action": ["ec2:Describe*", "ec2:AuthorizeSecurityGroupIngress"], "Resource": ["*"]},
        {
            "Effect": "Allow",
            "Action": ["ec2:CreateSecurityGroup"],
            "Resource": [
                f"arn:aws:ec2:*:{POLICY_ACCOUNT_PLACE_HOLDER}:security-group/*",
                f"arn:aws:ec2:*:{POLICY_ACCOUNT_PLACE_HOLDER}:vpc/*",
            ],
        },
        {
            "Effect": "Allow",
            "Action": ["ec2:CreateKeyPair"],
            "Resource": [f"arn:aws:ec2:*:{POLICY_ACCOUNT_PLACE_HOLDER}:key-pair/ag_ray_cluster*"],
        },
        {
            "Effect": "Allow",
            "Action": ["ec2:DeleteKeyPair"],
            "Resource": [f"arn:aws:ec2:*:{POLICY_ACCOUNT_PLACE_HOLDER}:key-pair/ag_ray_cluster*"],
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:GetInstanceProfile",
                "iam:CreateInstanceProfile",
                "iam:CreateRole",
                "iam:GetRole",
                "iam:AttachRolePolicy",
                "iam:DetachRolePolicy",
                "iam:AddRoleToInstanceProfile",
                "iam:PassRole",
            ],
            "Resource": ["*"],
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:CreatePolicy",
                "iam:DeletePolicy",
            ],
            "Resource": [f"arn:aws:iam::{POLICY_ACCOUNT_PLACE_HOLDER}:policy/AGRayClusterPolicy*"],
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:PutObjectAcl",
                "s3:GetObject",
                "s3:GetObjectAcl",
                "s3:AbortMultipartUpload",
            ],
            "Resource": [
                f"arn:aws:s3:::{POLICY_BUCKET_PLACE_HOLDER}/*",
                f"arn:aws:s3:::{POLICY_BUCKET_PLACE_HOLDER}",
                "arn:aws:s3:::*SageMaker*",
                "arn:aws:s3:::*Sagemaker*",
                "arn:aws:s3:::*sagemaker*",
            ],
        },
        {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": ["*"]},
        {
            "Effect": "Allow",
            "Action": ["iam:ListPolicies", "iam:ListEntitiesForPolicy", "iam:ListPolicyVersions"],
            "Resource": ["*"],
        },
    ],
}

ECR_READ_ONLY = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
