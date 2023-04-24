from .constants import CLOUD_RESOURCE_PREFIX, POLICY_ACCOUNT_PLACE_HOLDER, POLICY_BUCKET_PLACE_HOLDER

SAGEMAKER_TRUST_RELATIONSHIP_FILE_NAME = "ag_cloud_sagemaker_trust_relationship.json"
SAGEMAKER_IAM_POLICY_FILE_NAME = "ag_cloud_sagemaker_iam_policy.json"

SAGEMAKER_TRUST_RELATIONSHIP = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "",
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com",
                "AWS": f"arn:aws:iam::{POLICY_ACCOUNT_PLACE_HOLDER}:root",
            },
            "Action": "sts:AssumeRole",
        }
    ],
}

SAGEMAKER_CLOUD_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "SageMaker",
            "Effect": "Allow",
            "Action": [
                "sagemaker:DescribeEndpoint",
                "sagemaker:DescribeEndpointConfig",
                "sagemaker:DescribeModel",
                "sagemaker:DescribeTrainingJob",
                "sagemaker:DescribeTransformJob",
                "sagemaker:CreateArtifact",
                "sagemaker:CreateEndpoint",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:CreateModel",
                "sagemaker:CreateTrainingJob",
                "sagemaker:CreateTransformJob",
                "sagemaker:DeleteEndpoint",
                "sagemaker:DeleteEndpointConfig",
                "sagemaker:DeleteModel",
                "sagemaker:UpdateArtifact",
                "sagemaker:UpdateEndpoint",
                "sagemaker:InvokeEndpoint",
                "sagemaker:ListTags",  # Needed for re-attach job
            ],
            "Resource": [
                f"arn:aws:sagemaker:*:{POLICY_ACCOUNT_PLACE_HOLDER}:endpoint/{CLOUD_RESOURCE_PREFIX}*",
                f"arn:aws:sagemaker:*:{POLICY_ACCOUNT_PLACE_HOLDER}:endpoint-config/{CLOUD_RESOURCE_PREFIX}*",
                f"arn:aws:sagemaker:*:{POLICY_ACCOUNT_PLACE_HOLDER}:model/autogluon-inference*",
                f"arn:aws:sagemaker:*:{POLICY_ACCOUNT_PLACE_HOLDER}:training-job/{CLOUD_RESOURCE_PREFIX}*",
                f"arn:aws:sagemaker:*:{POLICY_ACCOUNT_PLACE_HOLDER}:transform-job/{CLOUD_RESOURCE_PREFIX}*",
            ],
        },
        {
            "Sid": "IAM",
            "Effect": "Allow",
            "Action": ["iam:PassRole"],
            "Resource": [
                f"arn:aws:iam::{POLICY_ACCOUNT_PLACE_HOLDER}:role/*",
            ],
        },
        {
            "Sid": "CloudWatchDescribe",
            "Effect": "Allow",
            "Action": ["logs:DescribeLogStreams"],
            "Resource": [f"arn:aws:logs:*:{POLICY_ACCOUNT_PLACE_HOLDER}:log-group:*"],
        },
        {
            "Sid": "CloudWatchGet",
            "Effect": "Allow",
            "Action": ["logs:GetLogEvents"],
            "Resource": [f"arn:aws:logs:*:{POLICY_ACCOUNT_PLACE_HOLDER}:log-group:*:log-stream:*"],
        },
        {
            "Sid": "S3Object",
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
        {
            "Effect": "Allow",
            "Action": [
                "s3:CreateBucket",
                "s3:GetBucketLocation",
                "s3:ListBucket",
                "s3:GetBucketCors",
                "s3:PutBucketCors",
                "s3:GetBucketAcl",
                "s3:PutObjectAcl",
            ],
            "Resource": [
                f"arn:aws:s3:::{POLICY_BUCKET_PLACE_HOLDER}/*",
                f"arn:aws:s3:::{POLICY_BUCKET_PLACE_HOLDER}",
                "arn:aws:s3:::*SageMaker*",
                "arn:aws:s3:::*Sagemaker*",
                "arn:aws:s3:::*sagemaker*",
            ],
        },
        {
            "Sid": "ListEvents",
            "Effect": "Allow",
            "Action": [
                "s3:ListAllMyBuckets",
                "sagemaker:ListEndpointConfigs",
                "sagemaker:ListEndpoints",
                "sagemaker:ListTransformJobs",
                "sagemaker:ListTrainingJobs",
                "sagemaker:ListModels",
                "sagemaker:ListDomains",
            ],
            "Resource": ["*"],
        },
        {
            "Effect": "Allow",
            "Action": "sagemaker:*",
            "Resource": ["arn:aws:sagemaker:*:*:flow-definition/*"],
            "Condition": {"StringEqualsIfExists": {"sagemaker:WorkteamType": ["private-crowd", "vendor-crowd"]}},
        },
        {
            "Sid": "Others",
            "Effect": "Allow",
            "Action": [
                "ecr:BatchGetImage",
                "ecr:Describe*",
                "ecr:GetAuthorizationToken",
                "ecr:GetDownloadUrlForLayer",
                "logs:CreateLogDelivery",
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:DeleteLogDelivery",
                "logs:Describe*",
                "logs:GetLogDelivery",
                "logs:GetLogEvents",
                "logs:ListLogDeliveries",
                "logs:PutLogEvents",
                "logs:PutResourcePolicy",
                "logs:UpdateLogDelivery",
            ],
            "Resource": ["*"],
        },
    ],
}
