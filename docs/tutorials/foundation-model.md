# Run Foundation Models with AutoGluon-Cloud

AutoGluon-Cloud lets you run pretrained foundation models on Amazon SageMaker — deploy real-time or serverless endpoints, or run batch predictions, with no dataset-specific training.

```{note}
This tutorial assumes you've already set up AutoGluon-Cloud on AWS. If you haven't, see [Setup](setup.md) first.
```

```{attention}
SageMaker compute and S3 storage are billed to your AWS account. AutoGluon-Cloud is a free wrapper, but it's your responsibility to monitor usage to avoid unexpected charges.
```

## What are Foundation Models?

Foundation models are large pretrained models that perform inference **zero-shot** — no fit step, no dataset-specific training. They've been trained on massive and diverse datasets, so they generalize to unseen data out of the box.

The standard CloudPredictor workflow follows a **fit → deploy / predict** pattern. Foundation models skip the fit step entirely. {py:class}`~autogluon.cloud.TimeSeriesFoundationModel` is the entry point: pick a model, call `predict()` or `deploy()`, done.

## Time Series Forecasting

### Available Models

| Model ID | Model Family | Covariates |
|----------|-------------|------------|
| `chronos-2` | Chronos-2 | ✅ |
| `chronos-bolt-tiny` | Chronos-Bolt | ❌ |
| `chronos-bolt-small` | Chronos-Bolt | ❌ |
| `chronos-bolt-base` | Chronos-Bolt | ❌ |

[Chronos-2](https://huggingface.co/autogluon/chronos-2) is the recommended model — it supports covariates, cross-learning across items, and context lengths up to 8192 time steps. For background on Chronos, see the [Forecasting with Chronos-2](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html) tutorial.

### Choosing an Inference Option

AutoGluon-Cloud supports three ways to run predictions, each with different cost and performance trade-offs:

1. **[Real-time endpoint](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)** — `model.deploy(inference_mode="realtime")`
    - ✅ Highest throughput, consistently low latency, supports both GPU and CPU instances
    - ✅ Simple setup
    - ❌ You pay for the time the endpoint is running (can be configured to [scale to zero](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-zero-instances.html))

2. **[Serverless endpoint (CPU only)](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)** — `model.deploy(inference_mode="serverless")`
    - ✅ Pay only for active inference time, no infrastructure management, scales to zero
    - ✅ Cost-efficient for intermittent or unpredictable traffic
    - ❌ Cold-start latency on first request after idle, lower throughput than realtime
    - ❌ CPU-only — not suitable for models that require a GPU

3. **Batch prediction** — `model.predict(...)`
    - ✅ Pay only for active compute time, no persistent infrastructure
    - ✅ Cost-efficient for large-scale prediction jobs
    - ❌ Job initialization takes a few minutes (not suitable for interactive use)

### Batch Prediction

```python
import pandas as pd
from autogluon.cloud import TimeSeriesFoundationModel

data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_tiny/train.csv")

# `known_covariates` is optional — pass it if you have future values of covariates available at prediction time
# (e.g., holidays, promotions, weather forecasts).
known_covariates = None

model = TimeSeriesFoundationModel("chronos-2", cloud_output_path="s3://YOUR-BUCKET/ag-foundation-model")

predictions = model.predict(
    data=data,
    prediction_length=24,
    target="target",
    id_column="item_id",
    timestamp_column="timestamp",
    known_covariates=known_covariates,
)

print(predictions.head())
```

The `predict()` method accepts the following key parameters:

- `data` — historical time series as a DataFrame or S3 path (long format)
- `prediction_length` — number of future time steps to predict
- `target`, `id_column`, `timestamp_column` — column names in `data`; defaults shown above
- `known_covariates` — DataFrame with future values of known covariates (see [Using Covariates with Chronos-2](#using-covariates-with-chronos-2))
- `quantile_levels` — (optional) list of quantiles to predict (e.g., `[0.1, 0.5, 0.9]`)
- `instance_type` — (optional) SageMaker instance type; defaults to the model registry value
- `wait` — (optional) if `False`, returns immediately with a future; call `.result()` to retrieve predictions later

The remaining examples in this tutorial assume the default column names.

### Real-Time Endpoint

For low-latency predictions, deploy the model as a SageMaker endpoint:

```python
model = TimeSeriesFoundationModel("chronos-2", cloud_output_path="s3://YOUR-BUCKET/ag-foundation-model")

# Deploy — this takes a few minutes
endpoint = model.deploy()

predictions = endpoint.predict(data=data, prediction_length=24)
print(predictions.head())
```

The endpoint stays active until you explicitly delete it:

```python
# Always clean up to avoid ongoing charges
endpoint.delete_endpoint()
```

### Serverless Endpoint

For sporadic or low-volume traffic, deploy a serverless endpoint instead. SageMaker provisions capacity on demand and scales to zero between requests:

```python
endpoint = model.deploy(inference_mode="serverless")
predictions = endpoint.predict(data=data, prediction_length=24)
endpoint.delete_endpoint()
```

The default configuration uses 4 GB of memory and a max concurrency of 5. Override via `inference_config`:

```python
endpoint = model.deploy(
    inference_mode="serverless",
    inference_config={"memory_size_in_mb": 6144, "max_concurrency": 10},
)
```

`inference_config` keys are forwarded to [`sagemaker.serverless.ServerlessInferenceConfig`](https://sagemaker.readthedocs.io/en/stable/api/inference/serverless.html).

### Using Covariates with Chronos-2

Chronos-2 natively supports two kinds of covariates — external variables that provide additional context for forecasting:

- **Known future covariates** — values are known at prediction time (e.g., holidays, promotions, weather forecasts). Pass via `known_covariates` (a DataFrame with `id_column`, `timestamp_column`, and one column per covariate).
- **Past covariates** — only observed historically (e.g., past sales of related products). Include them as additional columns in `data`.

`known_covariates` is accepted by both `model.predict(...)` and `endpoint.predict(...)`; covariate column names are inferred from the DataFrame columns (excluding `id_column` and `timestamp_column`). Both `data` and `known_covariates` accept DataFrames or S3 paths.
