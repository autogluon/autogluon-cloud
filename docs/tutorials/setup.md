# Set Up AutoGluon-Cloud on AWS

AutoGluon-Cloud needs two AWS resources to operate:

- An **IAM role** that SageMaker assumes to run training and inference jobs.
- An **S3 bucket** where training artifacts and trained models are stored.

The fastest way to set both up is the `autogluon-cloud bootstrap` command shipped with the package. If you already have a role and bucket, use `register` instead. This page walks through both paths and the day-2 commands (`status`, `teardown`).

## Install

```bash
pip install -U autogluon.cloud
```

This installs the `autogluon-cloud` CLI alongside the Python API.


## Quickstart: `bootstrap`

If you have AWS credentials configured (via `aws configure`, `AWS_*` env vars, SSO, or an instance profile), run:

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

This deploys a CloudFormation stack (`ag-cloud-sagemaker` by default), creates the IAM role and S3 bucket, and saves both to `~/.autogluon/cloud.yaml`. Subsequent `CloudPredictor` calls pick the saved values up automatically.

```{note}
Review the CloudFormation template before deploying: {repo-file}`src/autogluon/cloud/templates/ag_cloud_sagemaker.yaml`.
```


## Already have a role and bucket? Use `register`

If your platform team has provisioned an IAM role and S3 bucket for you, skip CloudFormation entirely and just tell AutoGluon-Cloud about them:

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

`register` makes no AWS calls — it only persists the values to `~/.autogluon/cloud.yaml`. The IAM role must trust `sagemaker.amazonaws.com` and have permissions equivalent to AWS's `AmazonSageMakerFullAccess` managed policy plus read/write access to your bucket.


## Check your setup: `status`

Verify the saved resources still exist and are accessible:

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

Each backend's bucket, role, and (if applicable) CloudFormation stack are checked. `ok` means the resource exists; `ok (unverified ...)` means the caller lacks the IAM permission to verify (the resource is probably fine, but `status` couldn't confirm).


## Tear down: `teardown`

When you're done with AutoGluon-Cloud and want to remove everything it created:

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

This deletes the CloudFormation stack(s) created by `bootstrap` and removes `~/.autogluon/cloud.yaml`. Backends added via `register` (no stack) only have their config entry removed — your existing role and bucket are left untouched.

```{warning}
CloudFormation refuses to delete non-empty S3 buckets. If your bucket holds training artifacts you want to discard, empty it first with `aws s3 rm s3://<bucket> --recursive`.
```


## Where the config lives

`bootstrap` and `register` both write to `~/.autogluon/cloud.yaml`. The file is keyed by backend, so you can have separate entries for different backends side by side. Override the directory with the `AG_CONFIG_DIR` environment variable.
