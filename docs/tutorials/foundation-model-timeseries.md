---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Run Pretrained Foundation Models on Amazon SageMaker

Foundation models are large pretrained models that generate predictions **zero-shot** on new data. Because they're trained on massive, diverse datasets, they generalize to unseen data out of the box — no dataset-specific fitting required.

That makes the workflow much simpler than [training your own time series predictor](./predictor-timeseries.md), which requires you to first fit a predictor on your data and then manage the trained artifact. With foundation models you skip the fit step entirely and go straight to deploying an endpoint or running batch predictions.

AutoGluon-Cloud exposes this workflow through {py:class}`~autogluon.cloud.TimeSeriesFoundationModel`. For now it covers time series forecasting only, with models like Chronos-2 available out of the box.

## Create the model

```{important}
Before running any code below, follow the [Setup tutorial](./setup.md) to register the IAM role and S3 bucket that SageMaker will use. The examples assume those resources are saved in `~/.autogluon/cloud.yaml`.
```

```python
from autogluon.cloud import TimeSeriesFoundationModel

model = TimeSeriesFoundationModel(model_id="chronos-2")
```

The rest of the tutorial reuses this `model` object.

### Available models

The following `model_id` values are currently supported. Chronos-2 models natively support covariates and cross-learning across items, while Chronos-Bolt is univariate-only.

| Model ID | Documentation | Weights |
|----------|---------------|---------|
| `chronos-2` | [Chronos2Model](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html#autogluon.timeseries.models.Chronos2Model) | [autogluon/chronos-2](https://huggingface.co/autogluon/chronos-2) |
| `chronos-2-small` | [Chronos2Model](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html#autogluon.timeseries.models.Chronos2Model) | [autogluon/chronos-2-small](https://huggingface.co/autogluon/chronos-2-small) |
| `chronos-bolt-tiny` | [ChronosModel](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html#autogluon.timeseries.models.ChronosModel) | [autogluon/chronos-bolt-tiny](https://huggingface.co/autogluon/chronos-bolt-tiny) |
| `chronos-bolt-small` | [ChronosModel](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html#autogluon.timeseries.models.ChronosModel) | [autogluon/chronos-bolt-small](https://huggingface.co/autogluon/chronos-bolt-small) |
| `chronos-bolt-base` | [ChronosModel](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html#autogluon.timeseries.models.ChronosModel) | [autogluon/chronos-bolt-base](https://huggingface.co/autogluon/chronos-bolt-base) |

`chronos-2` is the recommended model — it supports covariates, cross-learning across items, and context lengths up to 8192 time steps. For background on Chronos models, see the [Forecasting with Chronos-2](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html) tutorial.

## Data

The examples use a retail sales dataset with weekly sales for 1,115 stores. Load the historical observations:

```{code-cell} ipython3
import pandas as pd

data = pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/retail_sales/train.parquet")
data.head()
```

At a minimum, the input must contain three columns: an item ID, a timestamp, and the target value to forecast — here, `id`, `timestamp`, and `Sales`. The remaining columns (`Open`, `Promo`, `SchoolHoliday`, `StateHoliday`, `Customers`) are covariates, used by models that support them like Chronos-2 and ignored by univariate-only models like Chronos-Bolt. See the [Time Series Quick Start](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-quick-start.html) for the long-format schema.

Chronos-2 can optionally use future values of covariates known ahead of time (e.g. holidays or planned promotions). The test split contains those future values — drop `Sales` (the target) since it's what we want to predict:

```{code-cell} ipython3
known_covariates = (
    pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/retail_sales/test.parquet")
    .drop(columns=["Sales"])
)
known_covariates.head()
```

## Inference modes

{py:class}`~autogluon.cloud.TimeSeriesFoundationModel` supports three inference modes on SageMaker. The right choice depends on how often you need predictions and how much latency you can tolerate:

- **Batch prediction** — launch a one-off SageMaker job that scores a dataset and writes the results to S3. Compute spins up, runs, and shuts down automatically. Best for offline forecasting on larger datasets where minutes of startup latency are fine.
- **Real-time inference** — deploy the model to a long-running SageMaker endpoint and send requests over HTTPS. Lowest per-request latency, supports GPU instances. You pay for the endpoint as long as it's up, so best when you need predictions on demand and have steady traffic.
- **Serverless inference** — deploy to a SageMaker Serverless Inference endpoint that scales to zero between requests. You only pay for active inference time. Best for intermittent or unpredictable traffic. Trade-offs: CPU only, cold-start latency on the first request after idle, and an extra setup step to bundle weights into a single artifact.

The examples below all reuse the `data` and `known_covariates` DataFrames loaded above.

## Batch prediction

Use {py:meth}`~autogluon.cloud.TimeSeriesFoundationModel.predict` to score a dataset as a one-off job. It returns a DataFrame of forecasts:

```python
predictions = model.predict(
    data=data,
    target="Sales",
    id_column="id",
    timestamp_column="timestamp",
    prediction_length=13,
    known_covariates=known_covariates,  # optional
)
```

The job also writes the forecasts to S3 as a CSV. By default they land at `{cloud_output_path}/{job_name}/predictions.csv`; pass `predictions_path` to choose an explicit destination:

```python
predictions = model.predict(
    data=data,
    target="Sales",
    id_column="id",
    timestamp_column="timestamp",
    prediction_length=13,
    predictions_path="s3://my-bucket/forecasts/2026-06-02.csv",
)
```

For long-running jobs you can return immediately with `wait=False`. `predict()` then returns a `JobPredictionFuture` you can poll with `.status()` and resolve with `.result()`:

```python
future = model.predict(
    data=data,
    target="Sales",
    id_column="id",
    timestamp_column="timestamp",
    prediction_length=13,
    wait=False,
)

print(future.job_name, future.status())  # 'ag-...', 'InProgress'

predictions = future.result()  # blocks until the job finishes, returns a DataFrame
```

## Real-time inference

Deploy the model to a SageMaker endpoint with {py:meth}`~autogluon.cloud.TimeSeriesFoundationModel.deploy`, then send requests through the returned endpoint. Pick an `instance_type` based on cost and latency requirements (defaults to `ml.g5.xlarge`):

```python
endpoint = model.deploy(instance_type="ml.g5.xlarge")  # takes a few minutes

predictions = endpoint.predict(
    data=data,
    target="Sales",
    id_column="id",
    timestamp_column="timestamp",
    prediction_length=13,
    known_covariates=known_covariates,  # optional
)
```

The endpoint stays active — and billed — until you delete it:

```python
endpoint.delete_endpoint()
```

### Invoke the endpoint without AutoGluon-Cloud

The deployed endpoint is a normal SageMaker endpoint, so you can invoke it from any AWS SDK. Unlike the trained-predictor case, foundation model endpoints always need per-request inference settings (`prediction_length`, `freq`, `target`, etc.) bundled into the payload — so plain CSV is not supported. Pick one of the two structured payload formats below.

:::{dropdown} Payload formats — boto3 examples
:animate: fade-in-slide-down
:color: secondary

**Option 1: AutoGluon-Cloud's native `application/x-autogluon` envelope.** Each DataFrame is serialized as base64-encoded parquet, with inference settings carried in `inference_kwargs`. This is what {py:meth}`autogluon.cloud.TimeSeriesEndpoint.predict` sends under the hood:

```python
import base64
import io
import json
import boto3
import pandas as pd

def df_to_b64(df: pd.DataFrame) -> str:
    return base64.b64encode(df.to_parquet()).decode("ascii")

data = pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/retail_sales/train.parquet")
known_covariates = (
    pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/retail_sales/test.parquet")
    .drop(columns=["Sales"])
)

payload = {
    "version": 1,
    "data": df_to_b64(data),
    "known_covariates": df_to_b64(known_covariates),
    "inference_kwargs": {
        "prediction_length": 13,
        "target": "Sales",
        "id_column": "id",
        "timestamp_column": "timestamp",
    },
}

client = boto3.client("sagemaker-runtime")
response = client.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/x-autogluon",
    Accept="application/x-parquet",
    Body=json.dumps(payload).encode("utf-8"),
)
forecasts = pd.read_parquet(io.BytesIO(response["Body"].read()))
```

**Option 2: Per-item JSON.** Each item is a JSON object with its target history and, optionally, past and future values of covariates inline. This is the same payload schema used by [Chronos-2 on SageMaker JumpStart](https://github.com/amazon-science/chronos-forecasting/blob/v2.2.2/notebooks/deploy-chronos-to-amazon-sagemaker.ipynb), so it's a drop-in if you already have code talking to a JumpStart endpoint:

```python
import io
import json
import boto3
import pandas as pd

payload = {
    "inputs": [
        {
            "item_id": "store_1",
            "start": "2014-01-05",                  # ISO timestamp of the first target value
            "target": [123.0, 145.0, 167.0, ...],   # historical target values
            "past_covariates": {                    # past covariate values (same length as target)
                "Promo": [0, 1, 0, ...],
                "SchoolHoliday": [0, 0, 1, ...],
            },
            "future_covariates": {                  # future values over the forecast horizon (length = prediction_length)
                "Promo": [1, 0, ..., 1],
                "SchoolHoliday": [0, 1, ..., 0],
            },
        },
        # ... one entry per item
    ],
    "parameters": {
        "prediction_length": 13,
        "freq": "W",                                # required when "start" is set
    },
}

client = boto3.client("sagemaker-runtime")
response = client.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Accept="application/x-parquet",
    Body=json.dumps(payload).encode("utf-8"),
)
forecasts = pd.read_parquet(io.BytesIO(response["Body"].read()))
```
:::

### Reattaching to an existing endpoint

To send requests to an endpoint that's already running (e.g. from a previous session, or one a
teammate deployed), build a {py:class}`~autogluon.cloud.TimeSeriesEndpoint` directly from the
endpoint name:

```python
from autogluon.cloud import TimeSeriesEndpoint

endpoint = TimeSeriesEndpoint(endpoint_name="my-existing-endpoint")
```

Pass a configured `boto3.Session` to use a non-default AWS profile or region. The endpoint must
have been deployed via AutoGluon-Cloud, since the request payload format is AutoGluon-specific.

## Serverless inference

Serverless endpoints scale to zero between requests, so you only pay for active inference time. They run network-isolated, which means the model weights have to be bundled into a single `model.tar.gz` ahead of time rather than downloaded from HuggingFace at deploy time. Use {py:meth}`~autogluon.cloud.TimeSeriesFoundationModel.cache_model_artifact` to do this once, then deploy with `inference_mode="serverless"`:

```python
cached_model = model.cache_model_artifact("s3://YOUR-BUCKET/fm-cache")

print(cached_model.model_artifact_uri)  # 's3://YOUR-BUCKET/fm-cache/chronos-2/model.tar.gz'

endpoint = cached_model.deploy(inference_mode="serverless")

predictions = endpoint.predict(
    data=data,
    target="Sales",
    id_column="id",
    timestamp_column="timestamp",
    prediction_length=13,
    known_covariates=known_covariates,  # optional
)

endpoint.delete_endpoint()
```

Subsequent runs can skip `cache_model_artifact` by passing the bundled artifact straight to the constructor:

```python
model = TimeSeriesFoundationModel(
    model_id="chronos-2",
    model_artifact_uri="s3://YOUR-BUCKET/fm-cache/chronos-2/model.tar.gz",
)
endpoint = model.deploy(inference_mode="serverless")
```

## Choosing an instance type

Each inference mode has different sweet spots. Defaults work for most users — read on if you want to optimize for cost or throughput, or the default GPU instance type isn't available in your account or region.

### Real-time endpoints

Pick a GPU if you need low latency, a CPU if cost matters more or a GPU instance isn't available in the chosen region.

| Instance | Cost\* | Throughput\* | Notes |
|---|---|---|---|
| `ml.c6i.2xlarge` | 1× | 1× | CPU fallback. Cheap, broadly available. |
| `ml.g4dn.xlarge` | ~2× | ~5× | Budget GPU. |
| `ml.g5.xlarge` (default) | ~3× | ~10× | Lowest latency. |

\*Rough ratios relative to `ml.c6i.2xlarge`. Throughput measured on `chronos-2` with a 100-item × 2000-step context, 64-step horizon. Actual numbers depend on payload, region, and pricing.

**A few rules of thumb:**

- **Single-GPU is enough.** Going beyond `xlarge` on a GPU instance gives little benefit — chronos-2 doesn't saturate one GPU at typical batch sizes.
- **Newer GPU generations are faster** (`g6.xlarge`, `g6e.xlarge`); availability varies by region.
- **For CPU, scaling vCPUs helps.** `c6i.2xlarge → 4xlarge → 8xlarge` roughly halves latency at each step.
- **Slow deploy?** If a deploy takes much longer than ~6 min for GPU or ~4 min for CPU, the region is likely out of capacity for that instance type — try a different instance type or region.

### Batch prediction

Batch jobs don't care about latency, so default to CPU instances such as `ml.m5.2xlarge` (current default) or `ml.c6i.2xlarge`. Consider GPU only for very large datasets (>10M rows). Typical job startup overhead is ~4 min on CPU.

### Serverless endpoints

You don't choose an instance type — SageMaker manages CPU sizing automatically. Cold starts are typically ~30s. GPU is not available for serverless inference.

