# Set Up AutoGluon-Cloud on AWS

First, install the `autogluon.cloud` package:

```bash
pip install autogluon.cloud
```

AutoGluon-Cloud runs training and inference on Amazon SageMaker on your behalf. Every `CloudPredictor` or `FoundationModel` you create needs two AWS resources:

- an **IAM role** that SageMaker assumes to run training and inference jobs
- an **S3 bucket** to stage data and store trained models

```{attention}
SageMaker compute and S3 storage are billed to your AWS account. AutoGluon-Cloud is a free wrapper, but it's your responsibility to monitor usage and delete endpoints when no longer needed.
```

There are three ways to supply these resources — if you're unsure, start with option 1.

## 1. Create new resources with {func}`~autogluon.cloud.bootstrap`

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

## 2. Use existing resources with {func}`~autogluon.cloud.register`

Run this if you already have an IAM role and S3 bucket that you want to use with AutoGluon-Cloud. The values are saved under `~/.autogluon/cloud.yaml` for future calls.

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

The role must trust the `sagemaker.amazonaws.com` principal and grant the permissions AutoGluon-Cloud needs to run SageMaker jobs plus read/write access to your bucket — for example, a [SageMaker execution role](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html). For the exact set of permissions, see the {repo-file}`CloudFormation template <src/autogluon/cloud/templates/ag_cloud_sagemaker.yaml>` that {func}`~autogluon.cloud.bootstrap` uses. The `region` where the jobs are executed must match the bucket's region.

## 3. Pass resources on each call

Skip the saved config entirely and provide the role and bucket every time you create a `CloudPredictor` or `FoundationModel`.

```python
from autogluon.cloud import SageMakerConfig, TabularCloudPredictor

predictor = TabularCloudPredictor(
    cloud_output_path="s3://my-autogluon-bucket/output",
    backend=SageMakerConfig(
        role_arn="arn:aws:iam::222222222222:role/MyAutoGluonRole",
        region="us-east-1",
    ),
)
```

Useful for one-off scripts or when you need different roles and buckets per call. The same role and bucket requirements as option 2 apply.

## Share backend settings across workflows

{class}`~autogluon.cloud.SageMakerConfig` works with both cloud predictors and foundation models. It holds
the region, execution role, VPC, encryption keys, and resource tags. You can reuse it across objects;
each object gets its own backend, jobs, and endpoint state.

```python
from autogluon.cloud import SageMakerConfig, TabularCloudPredictor, TimeSeriesFoundationModel

backend = SageMakerConfig(
    region="us-east-1",
    role_arn="arn:aws:iam::222222222222:role/MyAutoGluonRole",
    vpc_config={"subnets": ["subnet-..."], "security_group_ids": ["sg-..."]},
    output_kms_key="arn:aws:kms:us-east-1:222222222222:key/...",
    tags={"team": "forecasting"},
)

predictor = TabularCloudPredictor(
    backend=backend,
    cloud_output_path="s3://my-autogluon-bucket/training",
)
model = TimeSeriesFoundationModel(
    "chronos-2",
    backend=backend,
    cloud_output_path="s3://my-autogluon-bucket/inference",
)
```

The role and region you set explicitly take precedence over the saved configuration. Leaving them
unset uses the existing saved-config and AWS-identity fallbacks. `backend="sagemaker"` is shorthand
for `backend=SageMakerConfig()`.

`output_kms_key` encrypts training artifacts, batch transform outputs, and repacked or cached model
artifacts in S3. `volume_kms_key` separately controls training, batch transform, and realtime endpoint
storage encryption; leave it unset for instances with local NVMe storage. Resource sizes,
container environment variables, spot training, and serverless settings remain arguments to the
individual `fit()`, `predict()`, and `deploy()` calls.

## Advanced provider settings

Use `backend_overrides` for SageMaker request fields without a named argument. It maps request names
(the boto3 SageMaker client methods) to request fields in the PascalCase format of the
[SageMaker API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/Welcome.html) and boto3:

```python
predictions = model.predict(
    data,
    prediction_length=24,
    backend_overrides={
        "create_training_job": {
            "RetryStrategy": {"MaximumRetryAttempts": 2},
        },
    },
)
```

Foundation-model predictions and predictor training use `create_training_job`. Predictor batch
transform uses `create_model` and `create_transform_job`. Deployment accepts `create_model`,
`production_variant`, `create_endpoint_config`, and `create_endpoint`.

Only requests used by the operation are accepted. Nested dictionaries merge recursively over the
generated request; other values, including lists, replace the generated value. Overrides take
precedence over backend settings and named arguments.

## Managing the saved config

Once {func}`~autogluon.cloud.bootstrap` or {func}`~autogluon.cloud.register` has written to `~/.autogluon/cloud.yaml`, you may want to check that the role and bucket are still healthy before a long training run, or clean everything up when you're done with AutoGluon-Cloud. Two helper commands cover both:

- {func}`~autogluon.cloud.status` checks that the saved role and bucket still exist and are accessible — handy after IAM or S3 changes.
- {func}`~autogluon.cloud.teardown` deletes the CloudFormation stack created by {func}`~autogluon.cloud.bootstrap` and clears the saved config. Resources registered via {func}`~autogluon.cloud.register` are left untouched, since you own them.

The config path can be overridden with the `AG_CONFIG_DIR` environment variable if you'd rather keep it somewhere other than `~/.autogluon/`.
