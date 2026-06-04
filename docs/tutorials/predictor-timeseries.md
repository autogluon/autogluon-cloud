# Train and Deploy a Time Series Predictor on Amazon SageMaker

```{note}
This tutorial covers time series forecasting. For tabular classification/regression, see [Train a Tabular Predictor](./predictor-tabular.md).
```

AutoGluon-Cloud lets you train, deploy, and run inference with AutoGluon time series predictors on AWS using the same APIs you'd use locally. Under the hood, it runs your jobs on [Amazon SageMaker](https://aws.amazon.com/sagemaker/) using AWS's official [AutoGluon deep learning containers](https://aws.github.io/deep-learning-containers/reference/available_images/#autogluon-training) — so you don't manage any infrastructure yourself.

## Training

```{important}
Before running any code below, follow the [Setup tutorial](setup.md) to register the IAM role and S3 bucket that SageMaker will use. The examples assume those resources are saved in `~/.autogluon/cloud.yaml`.
```

Create the predictor:

```python
from autogluon.cloud import TimeSeriesCloudPredictor

cloud_predictor = TimeSeriesCloudPredictor()
```

{py:meth}`TimeSeriesCloudPredictor.fit() <autogluon.cloud.TimeSeriesCloudPredictor.fit>` runs [`TimeSeriesPredictor.fit()`](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.fit.html) inside a remote SageMaker job — along with `train_data`, the `predictor_init_args` and `predictor_fit_args` are forwarded straight through. Training, model artifacts, and AutoGluon itself all live on the remote instance, so you don't need AutoGluon installed locally.

```python
cloud_predictor.fit(
    train_data="train.csv",  # DataFrame, local path, or S3 URL (CSV/Parquet)
    predictor_init_args={  # passed to TimeSeriesPredictor()
        "target": "target",
        "prediction_length": 24,
        "known_covariates_names": ["promo", "holiday"],
    },
    predictor_fit_args={"time_limit": 600},  # passed to TimeSeriesPredictor.fit()
    instance_type="ml.m5.2xlarge",
)
```

`train_data` can be a pandas DataFrame, or a path to a local or S3 file (CSV or Parquet). The data must be in **long format** with one row per `(item_id, timestamp)` pair plus a target column. See the [Time Series Quick Start](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-quick-start.html) for the expected schema and the [Forecasting In-Depth](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-indepth.html) tutorial for an overview of the different covariate types AutoGluon supports.

### Fit and predict in a single job

For workflows where fitting is light (e.g. fine-tuning a pretrained foundation model), {py:meth}`~autogluon.cloud.TimeSeriesCloudPredictor.fit_predict` runs both steps inside the same SageMaker job — saving the startup overhead of a second job. Predictions are generated against `train_data` and written to S3.

```python
forecasts = cloud_predictor.fit_predict(
    train_data="train.csv",
    predictor_init_args={
        "target": "target",
        "prediction_length": 24,
        "known_covariates_names": ["promo", "holiday"],
    },
    known_covariates="known_covariates.csv",  # required if known_covariates_names was set
    predictions_path="s3://my-bucket/forecasts/run-2026-06-02.csv",  # optional
)
```

By default predictions land at `{cloud_output_path}/{job_name}/predictions.csv`; pass `predictions_path` to choose a destination.

### Reattach to a training job
If your local connection drops, the training job keeps running on SageMaker. You can reattach with another `CloudPredictor` via {py:meth}`~autogluon.cloud.TimeSeriesCloudPredictor.attach_job` as long as you have the job name — it's logged when training starts (`INFO:sagemaker:Creating training-job with name: ag-cloudpredictor-...`) and also visible in the SageMaker console.

```python
another_cloud_predictor = TimeSeriesCloudPredictor()
another_cloud_predictor.attach_job(job_name="JOB_NAME")
```

A reattached job won't stream live logs — the full log becomes available once training finishes.

## Inference

Once a predictor is trained, you can get predictions in two ways:

- **Real-time inference**: deploy the predictor as a long-running SageMaker endpoint and send requests to it. Best when you need low-latency forecasts on demand — e.g. behind a user-facing service.
- **Batch inference**: launch a one-off SageMaker job that scores a dataset and writes the results to S3. Best for offline forecasting on larger datasets — compute spins up, runs, and shuts down automatically, so you only pay for what you use.

A rough guideline: if you need predictions less often than once an hour and can tolerate ~10 minutes of compute spin-up, batch inference is usually cheaper and easier to operate.

### Real-time inference

Deploy the predictor as a SageMaker endpoint with {py:meth}`~autogluon.cloud.TimeSeriesCloudPredictor.deploy`:

```python
cloud_predictor.deploy(
    instance_type="ml.m5.2xlarge",
)
```

Optionally, you can also attach to a deployed endpoint with {py:meth}`~autogluon.cloud.TimeSeriesCloudPredictor.attach_endpoint`:

```python
cloud_predictor.attach_endpoint(endpoint="ENDPOINT_NAME")
```

Send requests to the endpoint with {py:meth}`~autogluon.cloud.TimeSeriesCloudPredictor.predict_real_time`. It takes the historical observations to forecast from, plus optional `known_covariates` (required when `known_covariates_names` was set at fit time) and `static_features`. The result is a DataFrame with one row per `(item_id, future timestamp)` pair and a column for each predicted quantile (plus the `mean`):

```python
forecasts = cloud_predictor.predict_real_time(
    "train.csv",  # historical observations — forecasts start from the last timestamp per item
    known_covariates="known_covariates.csv",  # required if known_covariates_names was set
    static_features="static_features.csv",    # optional
)
#                            mean       0.1       0.5       0.9
# item_id timestamp
# 1       2015-05-03      28321.4   25103.2   28104.7   31682.1
#         2015-05-10      29014.9   25890.1   28911.5   32355.3
#         2015-05-17      19972.8   17612.6   19844.2   22463.7
# ...
```

Make sure you clean up the endpoint with {py:meth}`~autogluon.cloud.TimeSeriesCloudPredictor.cleanup_deployment`:

```python
cloud_predictor.cleanup_deployment()
```

To check whether an endpoint is currently attached, call {py:meth}`~autogluon.cloud.TimeSeriesCloudPredictor.info` and look for the `endpoint` key in the returned dict.

#### Invoke the endpoint without AutoGluon-Cloud
The deployed endpoint is a normal SageMaker endpoint, so you can invoke it from any AWS SDK. The simplest payload is the historical observations as CSV — forecasts are generated starting from the last timestamp of each item:

```python
import io
import boto3
import pandas as pd

train_data = pd.read_csv("train.csv")  # long format with item_id, timestamp, target

client = boto3.client("sagemaker-runtime")
response = client.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="text/csv",
    Accept="application/x-parquet",
    Body=train_data.to_csv(index=False),
)
forecasts = pd.read_parquet(io.BytesIO(response["Body"].read()))
```

The CSV format only carries the historical observations. To pass `static_features` or `known_covariates` (required when the predictor was fit with `known_covariates_names`), use one of the structured payload formats below.

:::{dropdown} Advanced payload formats — with static_features and known_covariates
:animate: fade-in-slide-down
:color: secondary

**Option 1: AutoGluon-Cloud's native `application/x-autogluon` envelope.** Each DataFrame is serialized as base64-encoded parquet and bundled in a single JSON object. This is what {py:meth}`~autogluon.cloud.TimeSeriesCloudPredictor.predict_real_time` sends under the hood:

```python
import base64
import io
import json
import boto3
import pandas as pd

def df_to_b64(df: pd.DataFrame) -> str:
    return base64.b64encode(df.to_parquet()).decode("ascii")

train_data = pd.read_csv("train.csv")
known_covariates = pd.read_csv("known_covariates.csv")
static_features = pd.read_csv("static_features.csv")  # optional

payload = {
    "version": 1,
    "data": df_to_b64(train_data),
    "known_covariates": df_to_b64(known_covariates),
    "static_features": df_to_b64(static_features),
    "inference_kwargs": {},  # prediction_length / quantile_levels are baked in at fit time
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
            "start": "2014-01-05",                            # ISO timestamp of the first target value
            "target": [123.0, 145.0, 167.0, ...],            # historical target values
            "past_covariates": {                              # past values of known_covariates (same length as target)
                "promo": [0, 1, 0, ...],
                "holiday": [0, 0, 1, ...],
            },
            "future_covariates": {                            # future values over the forecast horizon (length = prediction_length)
                "promo": [1, 0, ..., 1],
                "holiday": [0, 1, ..., 0],
            },
        },
        # ... one entry per item
    ],
    "parameters": {
        "prediction_length": 24,                              # must match the trained predictor's prediction_length
        "freq": "W",                                          # required when "start" is set
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

### Batch inference

To score a dataset as a one-off job, use {py:meth}`~autogluon.cloud.TimeSeriesCloudPredictor.predict`. Same kwargs as real-time — pass `known_covariates` (required when `known_covariates_names` was set at fit time) and `static_features` if relevant. It returns the same forecast DataFrame:

```python
forecasts = cloud_predictor.predict(
    "train.csv",  # historical observations — DataFrame, local path, or S3 URL (CSV/Parquet)
    known_covariates="known_covariates.csv",  # required if known_covariates_names was set
    static_features="static_features.csv",    # optional
    instance_type="ml.m5.2xlarge",
)
#                            mean       0.1       0.5       0.9
# item_id timestamp
# 1       2015-05-03      28321.4   25103.2   28104.7   31682.1
#         2015-05-10      29014.9   25890.1   28911.5   32355.3
# ...
```

## Inspect predictor state

To retrieve general info about a `CloudPredictor`, call {py:meth}`~autogluon.cloud.TimeSeriesCloudPredictor.info`:

```python
cloud_predictor.info()
```

It will output a dict similar to this:

```python
{
    'local_output_path': '/home/ubuntu/XXX/demo/AutogluonCloudPredictor/ag-20221111_174928',
    'cloud_output_path': 's3://XXX/timeseries-demo',
    'fit_job': {
        'name': 'ag-cloudpredictor-1668188968-e5c3',
        'status': 'Completed',
        'framework_version': '0.6.1',
        'artifact_path': 's3://XXX/timeseries-demo/model/ag-cloudpredictor-1668188968-e5c3/output/model.tar.gz'
    },
    'recent_transform_job': {
        'name': 'ag-cloudpredictor-1668189393-e95c',
        'status': 'Completed',
        'result_path': 's3://XXX/timeseries-demo/batch_transform/2022-11-11-17-56-33-991/results/test.parquet.out'
    },
    'transform_jobs': ['ag-cloudpredictor-1668189393-e95c'],
    'endpoint': 'ag-cloudpredictor-1668189208-d23b'
}
```

## Download the trained predictor
You can convert the `CloudPredictor` trained on SageMaker into a local AutoGluon predictor with {py:meth}`~autogluon.cloud.TimeSeriesCloudPredictor.to_local_predictor`, as long as you have the same version of AutoGluon installed locally.

```python
local_predictor = cloud_predictor.to_local_predictor(
    save_path="PATH"  # If not specified, CloudPredictor will create one.
)  # local_predictor would be a TimeSeriesPredictor
```

`to_local_predictor()` downloads the trained model tarball, expands it to your local disk, and loads it as the corresponding AutoGluon predictor.
