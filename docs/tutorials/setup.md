# Set Up AutoGluon-Cloud on AWS

AutoGluon-Cloud trains and deploys models on AWS SageMaker on your behalf. To do that, every `CloudPredictor` or `FoundationModel` you create needs two AWS resources:

- An **IAM role** that SageMaker assumes to run training and inference jobs.
- An **S3 bucket** to stage training data and store trained models.

You have two options for supplying them:

1. **Save them once** to `~/.autogluon/cloud.yaml`, and AutoGluon-Cloud will pick them up automatically on every call. This is the recommended path — set it up with [`bootstrap`](#bootstrap) or [`register`](#register) below.
2. **Pass them explicitly** to each `CloudPredictor` / `FoundationModel`, e.g. `CloudPredictor(role="arn:aws:iam::...", cloud_output_path="s3://my-bucket/...")`. Useful if you need different roles or buckets per call, or if you don't want a config file on disk.

The rest of this page covers option 1.

## Commands

AutoGluon-Cloud ships four commands for managing the saved configuration:

| Command | What it does | When to use it |
|---|---|---|
| [`bootstrap`](#bootstrap) | Provisions a role and bucket via CloudFormation, then saves them. | First-time setup with no existing AWS resources. |
| [`register`](#register) | Saves an existing role and bucket without provisioning anything. | Your platform team already gave you a role and bucket. |
| [`status`](#status) | Verifies the saved resources still exist and are accessible. | Sanity-check before training, or after IAM/S3 changes. |
| [`teardown`](#teardown) | Deletes resources created by `bootstrap` and the saved config. | Cleanup when you're done with AutoGluon-Cloud. |

Each command is available both as a CLI subcommand (`autogluon-cloud <command>`) and as a Python function (`from autogluon.cloud import <command>`). The sections below show both forms.

## Install

```bash
pip install -U autogluon.cloud
```

This installs the `autogluon-cloud` CLI alongside the Python API.


## `bootstrap`

Provisions an IAM role and S3 bucket via CloudFormation, then saves them to `~/.autogluon/cloud.yaml`. Use this if you don't already have AWS resources for AutoGluon-Cloud.

`bootstrap` uses the [standard boto3 credential resolution order](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#configuring-credentials) to find your AWS credentials, so anything that works for the AWS CLI or boto3 will work here (`aws configure`, `AWS_*` environment variables, an active SSO session, or an instance profile). Run:

::::{tab-set}
:::{tab-item} CLI
:sync: setup-cli
```bash
autogluon-cloud bootstrap
```
:::
:::{tab-item} Python
:sync: setup-py
```python
from autogluon.cloud import bootstrap

bootstrap()
```
:::
::::

The CloudFormation stack is named `ag-cloud-sagemaker` by default. Subsequent `CloudPredictor` calls pick the saved values up automatically.

```{note}
Review the CloudFormation template before deploying: {repo-file}`src/autogluon/cloud/templates/ag_cloud_sagemaker.yaml`.
```


## `register`

Tells AutoGluon-Cloud to use an IAM role and S3 bucket you already have. Use this when your platform team has provisioned them for you and you want to skip CloudFormation.

::::{tab-set}
:::{tab-item} CLI
:sync: setup-cli
```bash
autogluon-cloud register \
    --role arn:aws:iam::222222222222:role/MyAutoGluonRole \
    --bucket my-autogluon-bucket \
    --region us-east-1
```
:::
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
::::

`register` makes no AWS calls — it only persists the values to `~/.autogluon/cloud.yaml`. The IAM role must trust `sagemaker.amazonaws.com` and have permissions equivalent to AWS's [`AmazonSageMakerFullAccess`](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AmazonSageMakerFullAccess.html) managed policy plus read/write access to your bucket.


## `status`

Verifies that the saved IAM role, S3 bucket, and (if applicable) CloudFormation stack still exist and are accessible.

::::{tab-set}
:::{tab-item} CLI
:sync: setup-cli
```bash
autogluon-cloud status
```
:::
:::{tab-item} Python
:sync: setup-py
```python
from autogluon.cloud import status

reports = status()
```
:::
::::

`ok` means the resource exists; `ok (unverified ...)` means the caller lacks the IAM permission to verify (the resource is probably fine, but `status` couldn't confirm).


## `teardown`

Deletes the CloudFormation stacks created by `bootstrap` and removes `~/.autogluon/cloud.yaml`. Backends added via `register` only have their config entry removed — your existing role and bucket are left untouched.

::::{tab-set}
:::{tab-item} CLI
:sync: setup-cli
```bash
autogluon-cloud teardown
```
:::
:::{tab-item} Python
:sync: setup-py
```python
from autogluon.cloud import teardown

teardown()
```
:::
::::

```{warning}
CloudFormation refuses to delete non-empty S3 buckets. If your bucket holds training artifacts you want to discard, empty it first with `aws s3 rm s3://<bucket> --recursive`.
```


## Where the config lives

`bootstrap` and `register` both write to `~/.autogluon/cloud.yaml`. The file is keyed by backend, so you can have separate entries for different backends side by side. Override the directory with the `AG_CONFIG_DIR` environment variable.
