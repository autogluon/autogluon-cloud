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

# Run Pretrained Tabular Foundation Models on Amazon SageMaker

Tabular foundation models are large pretrained models that make predictions on a new dataset **in context**: you pass a set of labeled rows together with the rows to predict, and the model predicts them in a single forward pass. Because they're pretrained on large collections of diverse tables, they generalize to unseen tables out of the box — no hyperparameter tuning or lengthy training required.

That makes the workflow much simpler than [training your own tabular predictor](./predictor-tabular.md), which requires you to first fit a predictor on your data and then manage the trained artifact. With foundation models you skip the fit step entirely and go straight to running batch predictions or deploying an endpoint.

AutoGluon-Cloud exposes this workflow through {py:class}`~autogluon.cloud.TabularFoundationModel`, with models like Mitra, TabICLv2, TabDPT-Turbo and Nori available out of the box. For time series forecasting, see [Time Series Foundation Models](./foundation-model-timeseries.md).

## Create the model

```{important}
Before running any code below, follow the [Setup tutorial](./setup.md) to register the IAM role and S3 bucket that SageMaker will use. The examples assume those resources are saved in `~/.autogluon/cloud.yaml`.
```

```python
from autogluon.cloud import TabularFoundationModel

model = TabularFoundationModel(model_id="mitra-classifier")
```

The rest of the tutorial reuses this `model` object.

### Available models

The following `model_id` values are currently supported. Each `model_id` targets a single task — pick a `*-classifier` model for classification (binary or multiclass) and a `*-regressor` model for regression.

| Model ID | Task | Documentation | Weights |
|----------|------|---------------|---------|
| `mitra-classifier` | Classification | [MitraModel](https://auto.gluon.ai/stable/api/autogluon.tabular.models.html#autogluon.tabular.models.MitraModel) | [autogluon/mitra-classifier](https://huggingface.co/autogluon/mitra-classifier) |
| `mitra-regressor` | Regression | [MitraModel](https://auto.gluon.ai/stable/api/autogluon.tabular.models.html#autogluon.tabular.models.MitraModel) | [autogluon/mitra-regressor](https://huggingface.co/autogluon/mitra-regressor) |
| `tabicl-v2-classifier` | Classification | [TabICLModel](https://auto.gluon.ai/stable/api/autogluon.tabular.models.html#autogluon.tabular.models.TabICLModel) | [jingang/TabICL](https://huggingface.co/jingang/TabICL) |
| `tabicl-v2-regressor` | Regression | [TabICLModel](https://auto.gluon.ai/stable/api/autogluon.tabular.models.html#autogluon.tabular.models.TabICLModel) | [jingang/TabICL](https://huggingface.co/jingang/TabICL) |
| `tabdpt-turbo-classifier` | Classification | [TabDPTTurboModel](https://auto.gluon.ai/stable/api/autogluon.tabular.models.html#autogluon.tabular.models.TabDPTTurboModel) | [Layer6/TabDPT](https://huggingface.co/Layer6/TabDPT) |
| `tabdpt-turbo-regressor` | Regression | [TabDPTTurboModel](https://auto.gluon.ai/stable/api/autogluon.tabular.models.html#autogluon.tabular.models.TabDPTTurboModel) | [Layer6/TabDPT](https://huggingface.co/Layer6/TabDPT) |
| `nori-regressor` | Regression | [NoriModel](https://auto.gluon.ai/stable/api/autogluon.tabular.models.html#autogluon.tabular.models.NoriModel) | [Synthefy/Nori](https://huggingface.co/Synthefy/Nori) |
| `nori-30m-regressor` | Regression | [NoriModel](https://auto.gluon.ai/stable/api/autogluon.tabular.models.html#autogluon.tabular.models.NoriModel) | [Synthefy/Nori-30M](https://huggingface.co/Synthefy/Nori-30M) |

Mitra runs on CPU by default; the other models default to GPU instances. For background on tabular foundation models in AutoGluon, see the [Tabular Foundational Models](https://auto.gluon.ai/stable/tutorials/tabular/tabular-foundational-models.html) tutorial.

## Data

The examples use the [Adult Income](https://archive.ics.uci.edu/dataset/2/adult) dataset, where the task is to predict whether a person earns more than $50K a year. Tabular foundation models are designed for small to medium context sizes, so we take a random sample of the training set as the labeled context:

```{code-cell} ipython3
import pandas as pd

train_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv").sample(2000, random_state=0)
train_data.head()
```

The input is a regular tabular DataFrame: one row per example, one column per feature, plus the label column — here, `class`. The rows to predict must contain the same feature columns. Drop the label from the test split since it's what we want to predict:

```{code-cell} ipython3
test_data = (
    pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")
    .sample(500, random_state=0)
    .drop(columns=["class"])
)
test_data.head()
```

## Inference modes

{py:class}`~autogluon.cloud.TabularFoundationModel` supports two inference modes on SageMaker. In both, every call sends the labeled `train_data` as context along with the rows to predict, and AutoGluon fits a [`TabularPredictor`](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html) wrapping the foundation model on that context before predicting.

- **Batch prediction** — launch a one-off SageMaker job that scores a dataset and writes the results to S3. Compute spins up, runs, and shuts down automatically. Best for offline scoring of larger datasets where minutes of startup latency are fine.
- **Real-time inference** — deploy the model to a long-running SageMaker endpoint and send requests over HTTPS. Lowest per-request latency, supports GPU instances. You pay for the endpoint as long as it's up, so best when you need predictions on demand and have steady traffic.

```{note}
Serverless inference is not supported for tabular foundation models, since SageMaker Serverless Inference does not provide sufficient resources to run them.
```

The examples below all reuse the `train_data` and `test_data` DataFrames loaded above.

## Batch prediction

Use {py:meth}`~autogluon.cloud.TabularFoundationModel.predict` to score a dataset as a one-off job. It returns a Series of predictions:

```python
predictions = model.predict(
    test_data=test_data,
    train_data=train_data,
    label="class",
)
```

For classification, use {py:meth}`~autogluon.cloud.TabularFoundationModel.predict_proba` to get class probabilities. By default it returns both the predictions and the probabilities, computed in the same job:

```python
predictions, probabilities = model.predict_proba(
    test_data=test_data,
    train_data=train_data,
    label="class",
)
```

The job also writes the predictions to S3 as a CSV. By default they land at `{cloud_output_path}/{job_name}/predictions.csv`; pass `predictions_path` to choose an explicit destination:

```python
predictions = model.predict(
    test_data=test_data,
    train_data=train_data,
    label="class",
    predictions_path="s3://my-bucket/predictions/2026-06-02.csv",
)
```

For long-running jobs you can return immediately with `wait=False`. `predict()` then returns a `JobPredictionFuture` you can poll with `.status()` and resolve with `.result()`:

```python
future = model.predict(
    test_data=test_data,
    train_data=train_data,
    label="class",
    wait=False,
)

print(future.job_name, future.status())  # 'ag-...', 'InProgress'

predictions = future.result()  # blocks until the job finishes, returns a Series
```

## Real-time inference

Deploy the model to a SageMaker endpoint with {py:meth}`~autogluon.cloud.TabularFoundationModel.deploy`, then send requests through the returned {py:class}`~autogluon.cloud.TabularEndpoint`. Pick an `instance_type` based on cost and latency requirements (defaults to `ml.m5.4xlarge` for Mitra and `ml.g5.xlarge` for the other models):

```python
endpoint = model.deploy(instance_type="ml.m5.4xlarge")  # takes a few minutes

predictions = endpoint.predict(
    data=test_data,
    train_data=train_data,
    label="class",
)
predictions, probabilities = endpoint.predict_proba(
    data=test_data,
    train_data=train_data,
    label="class",
)
```

The endpoint holds no data between requests, so each request carries its own labeled context. This means you can use a single endpoint to serve predictions for different datasets and tasks of the same type — for example, a `mitra-classifier` endpoint can classify rows from any table, as long as each request includes the matching `train_data` and `label`.

Each request payload — `train_data` and `data` combined — is limited to 6 MB. For larger inputs, use [batch prediction](#batch-prediction) instead.

The endpoint stays active — and billed — until you delete it:

```python
endpoint.delete_endpoint()
```

### Invoke the endpoint without AutoGluon-Cloud

The deployed endpoint is a normal SageMaker endpoint, so you can invoke it from any AWS SDK. Unlike the trained-predictor case, foundation model endpoints need the labeled context and the label column name bundled into every request — so plain CSV is not supported. Use AutoGluon-Cloud's native `application/x-autogluon` envelope instead.

:::{dropdown} Payload format — boto3 example
:animate: fade-in-slide-down
:color: secondary

Each DataFrame is serialized as base64-encoded parquet, with the label column name carried in `inference_kwargs`. This is what {py:meth}`autogluon.cloud.TabularEndpoint.predict` sends under the hood:

```python
import base64
import io
import json
import boto3
import pandas as pd

def df_to_b64(df: pd.DataFrame) -> str:
    return base64.b64encode(df.to_parquet()).decode("ascii")

train_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv").sample(2000, random_state=0)
test_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv").drop(columns=["class"])

payload = {
    "version": 1,
    "data": df_to_b64(test_data),
    "train_data": df_to_b64(train_data),
    "inference_kwargs": {
        "label": "class",
    },
}

client = boto3.client("sagemaker-runtime")
response = client.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/x-autogluon",
    Accept="application/x-parquet",  # or "application/json", "text/csv"
    Body=json.dumps(payload).encode("utf-8"),
)
result = pd.read_parquet(io.BytesIO(response["Body"].read()))
```

For classification models, `result` contains the predicted label in the `class` column plus one `<class>_proba` column per class. For regression models, it contains a single column with the predictions.
:::

### Reattaching to an existing endpoint

To send requests to an endpoint that's already running (e.g. from a previous session, or one a
teammate deployed), build a {py:class}`~autogluon.cloud.TabularEndpoint` directly from the
endpoint name:

```python
from autogluon.cloud import TabularEndpoint

endpoint = TabularEndpoint(endpoint_name="my-existing-endpoint")
```

Pass a configured `boto3.Session` to use a non-default AWS profile or region. The endpoint must
have been deployed via AutoGluon-Cloud, since the request payload format is AutoGluon-specific.

## Choosing an instance type

Defaults work for most users — read on if you want to optimize for cost or throughput, or the default instance type isn't available in your account or region.

Mitra defaults to the CPU instance `ml.m5.4xlarge` for both batch prediction and real-time endpoints. TabICLv2, TabDPT-Turbo and Nori default to the GPU instance `ml.g5.xlarge`.

**A few rules of thumb:**

- **Context size drives cost.** Every request is processed together with the full `train_data`, so latency grows with the number of context rows and columns. Sample down the context if latency matters more than accuracy.
- **For CPU, scaling vCPUs helps.** Larger `m5` / `c6i` instances reduce latency at higher cost.
- **Slow deploy?** If a deploy takes much longer than ~6 min for GPU or ~4 min for CPU, the region is likely out of capacity for that instance type — try a different instance type or region.
