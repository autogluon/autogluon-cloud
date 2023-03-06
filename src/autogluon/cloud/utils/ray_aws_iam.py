from .constants import POLICY_ACCOUNT_PLACE_HOLDER

RAY_AWS_TRUST_RELATIONSHIP_FILE_NAME = "ag_cloud_ray_aws_trust_relationship.json"
RAY_AWS_IAM_POLICY_FILE_NAME = "ag_cloud_ray_aws_iam_policy.json"

RAY_AWS_TRUST_RELATIONSHIP = {
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}],
}

RAY_AWS_CLOUD_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {"Effect": "Allow", "Action": "ec2:RunInstances", "Resource": "arn:aws:ec2:*::image/ami-*"},
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
            "Resource": f"arn:aws:ec2:*:{POLICY_ACCOUNT_PLACE_HOLDER}:instance/*",
        },
        {"Effect": "Allow", "Action": ["ec2:Describe*", "ec2:AuthorizeSecurityGroupIngress"], "Resource": "*"},
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
            "Resource": f"arn:aws:ec2:*:{POLICY_ACCOUNT_PLACE_HOLDER}:key-pair/*",
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:GetInstanceProfile",
                "iam:CreateInstanceProfile",
                "iam:CreateRole",
                "iam:GetRole",
                "iam:AttachRolePolicy",
                "iam:AddRoleToInstanceProfile",
                "iam:PassRole",
            ],
            "Resource": ["*"],
        },
    ],
}
