# Time Series Foundation Models on Amazon SageMaker with AutoGluon-Cloud

AutoGluon-Cloud lets you run pretrained foundation models on Amazon SageMaker — deploy real-time endpoints or run batch predictions without any dataset-specific training.

## What are Foundation Models?

Foundation models are large pretrained models that can perform inference **zero-shot** on new data, without requiring a separate fit step. They have been trained on massive and diverse datasets, allowing them to generalize to unseen data out of the box.

With the standard CloudPredictor workflow, you follow a **fit → deploy → predict** pattern: train a task-specific model on your data, then deploy or run batch inference. Foundation models skip the fit step entirely — you can run predictions right away.

To support this workflow, AutoGluon-Cloud provides `TimeSeriesFoundationModel`: a new class that lets you go directly from model selection to inference. With it, you can:

- **Deploy a real-time endpoint** in minutes and start getting predictions immediately
- **Run batch predictions** on large datasets without waiting for a training job
- **Reduce costs** by eliminating the training step when zero-shot accuracy is sufficient

AutoGluon-Cloud currently supports foundation models for time series forecasting via `TimeSeriesFoundationModel`.

```{attention}
Costs for running cloud compute are managed by Amazon SageMaker, and storage costs are managed by AWS S3. AutoGluon-Cloud is a wrapper to these services at no additional charge. It is the user's responsibility to monitor compute usage and delete endpoints when no longer needed.
```

In the following, we will guide you step-by-step through using time series foundation models in AutoGluon-Cloud with the `TimeSeriesFoundationModel` class.

## Prerequisites

Install autogluon.cloud:

```bash
pip install autogluon.cloud
pip install -U sagemaker
```

You also need an IAM role with SageMaker permissions. See the [AutoGluon-Cloud tutorial](./autogluon-cloud.md) for setup instructions.

## Time Series Forecasting

### Available Models

The following table shows the time series foundation models currently supported by `TimeSeriesFoundationModel`. Some models, like Chronos-2, natively support covariates and cross-learning across items, while others are univariate-only.

| Model ID | Model Family | Covariates |
|----------|-------------|------------|
| `chronos-bolt-tiny` | Chronos-Bolt | ❌ |
| `chronos-bolt-small` | Chronos-Bolt | ❌ |
| `chronos-bolt-base` | Chronos-Bolt | ❌ |
| `chronos-2` | Chronos-2 | ✅ |

[Chronos-2](https://huggingface.co/autogluon/chronos-2) is the recommended model — it supports covariates, cross-learning across items, and context lengths up to 8192 time steps.

For background on Chronos models and their capabilities, see the [Forecasting with Chronos-2](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html) tutorial.

### Batch Prediction

The simplest way to get forecasts is with `predict()`. This launches a SageMaker batch transform job and returns predictions as a DataFrame.

```python
import pandas as pd
from autogluon.cloud import TimeSeriesFoundationModel

data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_tiny/train.csv")

model = TimeSeriesFoundationModel("chronos-2", cloud_output_path="s3://YOUR-BUCKET/ag-foundation-model")

predictions = model.predict(
    data=data,
    target="target",
    id_column="item_id",
    timestamp_column="timestamp",
    prediction_length=24,
)

print(predictions.head())
```

The `predict()` method accepts the following key parameters:

- `data` — historical time series as a DataFrame or S3 path (long format with item_id, timestamp, target columns)
- `target` — name of the column to forecast
- `prediction_length` — number of future time steps to predict
- `quantile_levels` — (optional) list of quantiles to predict (e.g., `[0.1, 0.5, 0.9]`)
- `instance_type` — (optional) SageMaker instance type; defaults to `ml.m5.2xlarge`

### Real-Time Endpoint

For low-latency predictions, deploy the model as a SageMaker endpoint:

```python
from autogluon.cloud import TimeSeriesFoundationModel

model = TimeSeriesFoundationModel("chronos-2", cloud_output_path="s3://YOUR-BUCKET/ag-foundation-model")

# Deploy — this takes a few minutes
endpoint = model.deploy()

# Make predictions
predictions = endpoint.predict(
    data=data,
    prediction_length=24,
    target="target",
    id_column="item_id",
    timestamp_column="timestamp",
)

print(predictions.head())
```

The endpoint stays active until you explicitly delete it:

```python
# Always clean up to avoid ongoing charges
endpoint.delete_endpoint()
```

The `deploy()` method accepts:

- `instance_type` — (optional) defaults to `ml.g5.xlarge` (GPU instance for faster inference)
- `endpoint_name` — (optional) custom name for the endpoint
- `hyperparameters` — (optional) model hyperparameters to override defaults

### Using Covariates with Chronos-2

Chronos-2 natively supports covariates — external variables that provide additional context for forecasting. Specifically, it supports:

- **Known future covariates** — variables whose future values are known at prediction time (e.g., holidays, promotions, weather forecasts)
- **Past covariates** — variables that are only observed historically (e.g., past sales of related products)

Pass known covariates via the `known_covariates` parameter in batch mode or through the endpoint:

```python
# Batch prediction with covariates
predictions = model.predict(
    data=data,
    target="target",
    id_column="item_id",
    timestamp_column="timestamp",
    prediction_length=24,
    known_covariates=future_covariates_df,  # DataFrame with id_column, timestamp_column, and covariate columns
)
```

```python
# Real-time endpoint with covariates
predictions = endpoint.predict(
    data=data,
    prediction_length=24,
    target="target",
    id_column="item_id",
    timestamp_column="timestamp",
    known_covariates=future_covariates_df,
)
```

The covariate column names are automatically inferred from the DataFrame columns (excluding `id_column` and `timestamp_column`). Both `data` and `known_covariates` can be passed as DataFrames or S3 paths.
