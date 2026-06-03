# Set Up AutoGluon-Cloud on AWS

First, install the `autogluon.cloud` package:

```bash
pip install autogluon.cloud
```

AutoGluon-Cloud runs training and inference on Amazon SageMaker on your behalf. Every `CloudPredictor` or `FoundationModel` you create needs two AWS resources:

- an **IAM role** that SageMaker assumes to run training and inference jobs
- an **S3 bucket** to stage data and store trained models

There are three ways to supply them — if you're unsure, start with option 1.

### 1. Create new resources with [`bootstrap`](../api/autogluon.cloud.bootstrap.rst)

Run this if you don't yet have an IAM role and S3 bucket set up for SageMaker. The role and bucket are provisioned on your account from a {repo-file}`CloudFormation template <src/autogluon/cloud/templates/ag_cloud_sagemaker.yaml>` and saved under `~/.autogluon/cloud.yaml` for future calls.

::::{tab-set}
:::{tab-item} Python
:sync: setup-py
```python
from autogluon.cloud import bootstrap

bootstrap()
```
:::
:::{tab-item} CLI
:sync: setup-cli
```bash
autogluon-cloud bootstrap
```
:::
::::

The CloudFormation stack is named `ag-cloud-sagemaker`. Your active AWS credentials are resolved in the [standard boto3 order](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#configuring-credentials).

### 2. Use existing resources with [`register`](../api/autogluon.cloud.register.rst)

Run this if you already have an IAM role and S3 bucket that you want to use with AutoGluon-Cloud. The values are saved under `~/.autogluon/cloud.yaml` for future calls. Makes no AWS calls.

::::{tab-set}
:::{tab-item} Python
:sync: setup-py
```python
from autogluon.cloud import register

register(
    role="arn:aws:iam::222222222222:role/MyAutoGluonRole",
    bucket="my-autogluon-bucket",
    region="us-east-1",
)
```
:::
:::{tab-item} CLI
:sync: setup-cli
```bash
autogluon-cloud register \
    --role arn:aws:iam::222222222222:role/MyAutoGluonRole \
    --bucket my-autogluon-bucket \
    --region us-east-1
```
:::
::::

The role must trust the `sagemaker.amazonaws.com` principal and have permissions equivalent to AWS's [`AmazonSageMakerFullAccess`](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AmazonSageMakerFullAccess.html) managed policy plus read/write access to your bucket. The `region` must match the bucket's region.

### 3. Pass resources on each call

Skip the saved config entirely and provide the role and bucket every time you create a `CloudPredictor` or `FoundationModel`.

```python
from autogluon.cloud import TabularCloudPredictor

predictor = TabularCloudPredictor(
    cloud_output_path="s3://my-autogluon-bucket/output",
    role="arn:aws:iam::222222222222:role/MyAutoGluonRole",
)
```

Useful for one-off scripts or when you need different roles and buckets per call. The same role and bucket requirements as option 2 apply.

## Managing the saved config

Once `bootstrap` or `register` has written to `~/.autogluon/cloud.yaml`, you may want to check that the role and bucket are still healthy before a long training run, or clean everything up when you're done with AutoGluon-Cloud. Two helper commands cover both:

- [`status`](../api/autogluon.cloud.status.rst) checks that the saved role and bucket still exist and are accessible — handy after IAM or S3 changes.
- [`teardown`](../api/autogluon.cloud.teardown.rst) deletes the CloudFormation stack created by `bootstrap` and clears the saved config. Resources registered via `register` are left untouched, since you own them.

The config path can be overridden with the `AG_CONFIG_DIR` environment variable if you'd rather keep it somewhere other than `~/.autogluon/`.
