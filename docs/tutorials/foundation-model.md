# Run Pretrained Foundation Models on Amazon SageMaker

Foundation models are large pretrained models that generate predictions **zero-shot** on new data. Because they're trained on massive, diverse datasets, they generalize to unseen data out of the box — no dataset-specific fitting required.

That makes the workflow much simpler than the [standard CloudPredictor workflow](./cloud-predictor.md), which requires you to first fit a predictor on your data and then manage the trained artifact. With foundation models you skip the fit step entirely and go straight to deploying an endpoint or running batch predictions.

AutoGluon-Cloud exposes this workflow through {py:class}`~autogluon.cloud.TimeSeriesFoundationModel`. For now it covers time series forecasting only, with models like Chronos-2 available out of the box.

```{attention}
SageMaker compute and S3 storage are billed to your AWS account. AutoGluon-Cloud is a free wrapper, but it's your responsibility to monitor usage and delete endpoints when no longer needed.
```

## Create the model

A {py:class}`~autogluon.cloud.TimeSeriesFoundationModel` needs an IAM execution role (so SageMaker can run jobs on your behalf) and an S3 bucket (to stage data and store outputs). There are two ways to supply them:

- Use a saved config (recommended). Save the role and bucket once to `~/.autogluon/cloud.yaml` — see [Setup](./setup.md) — and subsequent constructor calls will pick them up automatically:

  ```python
  from autogluon.cloud import TimeSeriesFoundationModel

  model = TimeSeriesFoundationModel(model_id="chronos-2")
  ```

- Pass them at construction. Useful when you need different roles or buckets per call:

  ```python
  model = TimeSeriesFoundationModel(
      model_id="chronos-2",
      role="arn:aws:iam::222222222222:role/MyAutoGluonRole",
      cloud_output_path="s3://my-autogluon-bucket/ag-foundation-model",
  )
  ```

The examples in the rest of this tutorial reuse a single `model` object created this way.

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

## Inference modes

{py:class}`~autogluon.cloud.TimeSeriesFoundationModel` supports three inference modes on SageMaker. The right choice depends on how often you need predictions and how much latency you can tolerate:

- **Batch prediction** — launch a one-off SageMaker job that scores a dataset and writes the results to S3. Compute spins up, runs, and shuts down automatically. Best for offline forecasting on larger datasets where minutes of startup latency are fine.
- **Real-time inference** — deploy the model to a long-running SageMaker endpoint and send requests over HTTPS. Lowest per-request latency, supports GPU instances. You pay for the endpoint as long as it's up, so best when you need predictions on demand and have steady traffic.
- **Serverless inference** — deploy to a SageMaker Serverless Inference endpoint that scales to zero between requests. You only pay for active inference time. Best for intermittent or unpredictable traffic. Trade-offs: CPU only, cold-start latency on the first request after idle, and an extra setup step to bundle weights into a single artifact.

Throughout the examples below, `data` is a pandas DataFrame in long format (one row per `(item_id, timestamp)` pair, plus a `target` column). See the [Time Series Quick Start](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-quick-start.html) for the expected format.

## Batch prediction

Use {py:meth}`~autogluon.cloud.TimeSeriesFoundationModel.predict` to score a dataset as a one-off job. It returns a DataFrame of forecasts:

```python
import pandas as pd

data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_tiny/train.csv")

predictions = model.predict(
    data=data,
    target="target",
    id_column="item_id",
    timestamp_column="timestamp",
    prediction_length=24,
)
```

To use covariates with Chronos-2, pass an optional `known_covariates` DataFrame containing future values of the covariate columns over the forecast horizon. See the [Forecasting In-Depth](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-indepth.html) tutorial for the expected covariates format.

```python
predictions = model.predict(
    data=data,
    target="target",
    id_column="item_id",
    timestamp_column="timestamp",
    prediction_length=24,
    known_covariates=future_covariates_df,
)
```

## Real-time inference

Deploy the model to a SageMaker endpoint with {py:meth}`~autogluon.cloud.TimeSeriesFoundationModel.deploy`, then send requests through the returned endpoint:

```python
endpoint = model.deploy()  # takes a few minutes

predictions = endpoint.predict(
    data=data,
    target="target",
    id_column="item_id",
    timestamp_column="timestamp",
    prediction_length=24,
    known_covariates=future_covariates_df,  # optional
)
```

The endpoint stays active — and billed — until you delete it:

```python
endpoint.delete_endpoint()
```

## Serverless inference

Serverless endpoints scale to zero between requests, so you only pay for active inference time. They run network-isolated, which means the model weights have to be bundled into a single `model.tar.gz` ahead of time rather than downloaded from HuggingFace at deploy time. Use {py:meth}`~autogluon.cloud.TimeSeriesFoundationModel.cache_model_artifact` to do this once, then deploy with `inference_mode="serverless"`:

```python
# Bundle weights to S3 once; reusable across deploys
cached_model = model.cache_model_artifact("s3://YOUR-BUCKET/fm-cache")

endpoint = cached_model.deploy(inference_mode="serverless")

predictions = endpoint.predict(
    data=data,
    target="target",
    id_column="item_id",
    timestamp_column="timestamp",
    prediction_length=24,
)

endpoint.delete_endpoint()
```
