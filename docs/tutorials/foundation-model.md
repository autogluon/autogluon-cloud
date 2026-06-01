# Time Series Foundation Models on Amazon SageMaker with AutoGluon-Cloud

AutoGluon-Cloud lets you run pretrained foundation models on Amazon SageMaker — deploy real-time endpoints or run batch predictions without any dataset-specific training.

## What are Foundation Models?

Foundation models are large pretrained models that can perform inference **zero-shot** on new data, without requiring a separate fit step. They have been trained on massive and diverse datasets, allowing them to generalize to unseen data out of the box.

With the standard CloudPredictor workflow, you follow a **fit → deploy / predict** pattern: first fit a predictor on your data, then deploy or run batch inference. Foundation models skip the fit step entirely — you can run predictions right away.

To support this workflow, AutoGluon-Cloud provides {py:class}`~autogluon.cloud.TimeSeriesFoundationModel`: a new class that lets you go directly from model selection to inference. With it, you can:

- **Deploy a real-time or serverless endpoint** in minutes and start getting predictions immediately
- **Run batch predictions** on large datasets without provisioning a persistent endpoint
- **Reduce costs** by eliminating the training step when zero-shot accuracy is sufficient

AutoGluon-Cloud currently supports foundation models for time series forecasting via {py:class}`~autogluon.cloud.TimeSeriesFoundationModel`.

```{attention}
Costs for running cloud compute are managed by Amazon SageMaker, and storage costs are managed by AWS S3. AutoGluon-Cloud is a wrapper to these services at no additional charge. It is the user's responsibility to monitor compute usage and delete endpoints when no longer needed.
```

In the following, we will guide you step-by-step through using time series foundation models in AutoGluon-Cloud with the {py:class}`~autogluon.cloud.TimeSeriesFoundationModel` class.

## Prerequisites

Install autogluon.cloud:

```bash
pip install autogluon.cloud
```

You also need an IAM role with SageMaker permissions. See the [AutoGluon-Cloud tutorial](./autogluon-cloud.md) for setup instructions.

## Time Series Forecasting

### Available Models

The following table shows the time series foundation models currently supported by {py:class}`~autogluon.cloud.TimeSeriesFoundationModel`. Some models, like Chronos-2, natively support covariates and cross-learning across items, while others are univariate-only.

| Model ID | Model Family | Covariates |
|----------|-------------|------------|
| `chronos-2` | Chronos-2 | ✅ |
| `chronos-bolt-tiny` | Chronos-Bolt | ❌ |
| `chronos-bolt-small` | Chronos-Bolt | ❌ |
| `chronos-bolt-base` | Chronos-Bolt | ❌ |

[Chronos-2](https://huggingface.co/autogluon/chronos-2) is the recommended model — it supports covariates, cross-learning across items, and context lengths up to 8192 time steps.

For background on Chronos models and their capabilities, see the [Forecasting with Chronos-2](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html) tutorial.

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
    - ❌ Requires data to be uploaded to S3 (autogluon-cloud handles this for you)

The examples below assume your data uses the default column names (`item_id`, `timestamp`, `target`). If your columns differ, pass `id_column=`, `timestamp_column=`, and `target=` to the relevant call.

### Batch Prediction

```python
import pandas as pd
from autogluon.cloud import TimeSeriesFoundationModel

data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_tiny/train.csv")

model = TimeSeriesFoundationModel("chronos-2", cloud_output_path="s3://YOUR-BUCKET/ag-foundation-model")

predictions = model.predict(
    data=data,
    prediction_length=24,
    # If your data uses different column names, pass them here:
    # id_column="my_id", timestamp_column="my_timestamp", target="my_target",
)

print(predictions.head())
```

The `predict()` method accepts the following key parameters:

- `data` — historical time series as a DataFrame or S3 path (long format)
- `prediction_length` — number of future time steps to predict
- `quantile_levels` — (optional) list of quantiles to predict (e.g., `[0.1, 0.5, 0.9]`)
- `instance_type` — (optional) SageMaker instance type; defaults to the model registry value
- `wait` — (optional) if `False`, returns immediately with a future; call `.result()` to retrieve predictions later

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

`inference_config` keys are forwarded to [`sagemaker.serverless.ServerlessInferenceConfig`](https://sagemaker.readthedocs.io/en/stable/api/inference/serverless.html); invalid keys raise `TypeError`.

### Using Covariates with Chronos-2

Chronos-2 natively supports covariates — external variables that provide additional context for forecasting. Specifically, it supports:

- **Known future covariates** — variables whose future values are known at prediction time (e.g., holidays, promotions, weather forecasts)
- **Past covariates** — variables that are only observed historically (e.g., past sales of related products)

Pass known covariates via the `known_covariates` parameter:

```python
predictions = model.predict(
    data=data,
    prediction_length=24,
    known_covariates=future_covariates_df,  # DataFrame with id_column, timestamp_column, and covariate columns
)
```

The same parameter works on a deployed endpoint:

```python
predictions = endpoint.predict(
    data=data,
    prediction_length=24,
    known_covariates=future_covariates_df,
)
```

The covariate column names are automatically inferred from the DataFrame columns (excluding `id_column` and `timestamp_column`). Both `data` and `known_covariates` can be passed as DataFrames or S3 paths.
