# Train and Deploy a Tabular Predictor on Amazon SageMaker

```{note}
This tutorial covers tabular classification and regression. For time series forecasting, see [Train a Time Series Predictor](./predictor-timeseries.md).
```

AutoGluon-Cloud lets you train, deploy, and run inference with AutoGluon tabular predictors on AWS using the same APIs you'd use locally. Under the hood, it runs your jobs on [Amazon SageMaker](https://aws.amazon.com/sagemaker/) using AWS's official [AutoGluon deep learning containers](https://aws.github.io/deep-learning-containers/reference/available_images/#autogluon-training) — so you don't manage any infrastructure yourself.

## Training

```{important}
Before running any code below, follow the [Setup tutorial](setup.md) to register the IAM role and S3 bucket that SageMaker will use. The examples assume those resources are saved in `~/.autogluon/cloud.yaml`.
```

Create the predictor:

```python
from autogluon.cloud import TabularCloudPredictor

cloud_predictor = TabularCloudPredictor()
```

{py:meth}`~autogluon.cloud.TabularCloudPredictor.fit` runs [`TabularPredictor.fit()`](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html) inside a remote SageMaker job — along with `train_data`, the `predictor_init_args` and `predictor_fit_args` are forwarded straight through. Training, model artifacts, and AutoGluon itself all live on the remote instance, so you don't need AutoGluon installed locally.

`train_data` can be a pandas DataFrame, or a path to a local or S3 file (CSV or Parquet). In every case AutoGluon-Cloud loads the data locally and uploads it to your `cloud_output_path` bucket before kicking off the SageMaker job.

```python
cloud_predictor.fit(
    train_data="train.csv",  # DataFrame, local path, or S3 URL (CSV/Parquet)
    predictor_init_args={"label": "label"},  # passed to TabularPredictor()
    predictor_fit_args={"time_limit": 120},  # passed to TabularPredictor.fit()
    instance_type="ml.m5.2xlarge",
)
```

### Reattach to a training job
If your local connection drops, the training job keeps running on SageMaker. You can reattach with another `CloudPredictor` via {py:meth}`~autogluon.cloud.TabularCloudPredictor.attach_job` as long as you have the job name — it's logged when training starts (`INFO:sagemaker:Creating training-job with name: ag-cloudpredictor-...`) and also visible in the SageMaker console.

```python
another_cloud_predictor = TabularCloudPredictor()
another_cloud_predictor.attach_job(job_name="JOB_NAME")
```

A reattached job won't stream live logs — the full log becomes available once training finishes.

## Inference

Once a predictor is trained, you can get predictions in two ways:

- **Real-time inference**: deploy the predictor as a long-running SageMaker endpoint and send requests to it. Best when you need low-latency predictions on demand — e.g. behind a user-facing service.
- **Batch inference**: launch a one-off SageMaker job that scores a dataset and writes the results to S3. Best for offline scoring of larger datasets — compute spins up, runs, and shuts down automatically, so you only pay for what you use.

A rough guideline: if you need predictions less often than once an hour and can tolerate ~10 minutes of compute spin-up, batch inference is usually cheaper and easier to operate.

### Real-time inference

Deploy the predictor as a SageMaker endpoint with {py:meth}`~autogluon.cloud.TabularCloudPredictor.deploy`:

```python
cloud_predictor.deploy(
    instance_type="ml.m5.2xlarge",
)
```

Optionally, you can also attach to a deployed endpoint with {py:meth}`~autogluon.cloud.TabularCloudPredictor.attach_endpoint`:

```python
cloud_predictor.attach_endpoint(endpoint="ENDPOINT_NAME")
```

Send requests to the endpoint with {py:meth}`~autogluon.cloud.TabularCloudPredictor.predict_real_time`, which returns a pandas Series of predictions:

```python
result = cloud_predictor.predict_real_time("test.csv")  # DataFrame, local path, or S3 URL
# 0      dog
# 1      cat
# 2      cat
# Name: label, dtype: object
```

For class probabilities, use {py:meth}`~autogluon.cloud.TabularCloudPredictor.predict_proba_real_time`, which returns a DataFrame with one column per class:

```python
result = cloud_predictor.predict_proba_real_time("test.csv")
#         dog       cat
# 0  0.682754  0.317246
# 1  0.195782  0.804218
# 2  0.372283  0.627717
```

Make sure you clean up the endpoint with {py:meth}`~autogluon.cloud.TabularCloudPredictor.cleanup_deployment`:

```python
cloud_predictor.cleanup_deployment()
```

To check whether an endpoint is currently attached, call {py:meth}`~autogluon.cloud.TabularCloudPredictor.info` and look for the `endpoint` key in the returned dict.

#### Invoke the endpoint without AutoGluon-Cloud
The deployed endpoint is a normal SageMaker endpoint, and you can invoke it through other methods. For example, to invoke it with boto3 directly:

```python
import boto3

client = boto3.client('sagemaker-runtime')
response = client.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType='text/csv',
    Accept='application/json',
    Body=test_data.to_csv()
)

#: Print the model endpoint's output.
print(response['Body'].read().decode())
```

### Batch inference

To score a dataset as a one-off job, use {py:meth}`~autogluon.cloud.TabularCloudPredictor.predict`. It returns a pandas Series of predictions:

```python
result = cloud_predictor.predict(
    "test.csv",  # DataFrame, local path, or S3 URL (CSV/Parquet)
    instance_type="ml.m5.2xlarge",
)
# 0      dog
# 1      cat
# 2      cat
# Name: label, dtype: object
```

For class probabilities, use {py:meth}`~autogluon.cloud.TabularCloudPredictor.predict_proba`. With `include_predict=True` (the default) it returns a `(predictions, probabilities)` tuple — useful because it avoids the cost of a second batch job. Pass `include_predict=False` to get the probabilities DataFrame alone:

```python
predictions, probabilities = cloud_predictor.predict_proba(
    "test.csv",
    include_predict=True,
    instance_type="ml.m5.2xlarge",
)
# predictions:
# 0      dog
# 1      cat
# 2      cat
# Name: label, dtype: object
#
# probabilities:
#         dog       cat
# 0  0.682754  0.317246
# 1  0.195782  0.804218
# 2  0.372283  0.627717
```

## Inspect predictor state

To retrieve general info about a `CloudPredictor`, call {py:meth}`~autogluon.cloud.TabularCloudPredictor.info`:

```python
cloud_predictor.info()
```

It will output a dict similar to this:

```python
{
    'local_output_path': '/home/ubuntu/XXX/demo/AutogluonCloudPredictor/ag-20221111_174928',
    'cloud_output_path': 's3://XXX/tabular-demo',
    'fit_job': {
        'name': 'ag-cloudpredictor-1668188968-e5c3',
        'status': 'Completed',
        'framework_version': '0.6.1',
        'artifact_path': 's3://XXX/tabular-demo/model/ag-cloudpredictor-1668188968-e5c3/output/model.tar.gz'
    },
    'recent_transform_job': {
        'name': 'ag-cloudpredictor-1668189393-e95c',
        'status': 'Completed',
        'result_path': 's3://XXX/tabular-demo/batch_transform/2022-11-11-17-56-33-991/results/test.csv.out'
    },
    'transform_jobs': ['ag-cloudpredictor-1668189393-e95c'],
    'endpoint': 'ag-cloudpredictor-1668189208-d23b'
}
```

## Download the trained predictor
You can convert the `CloudPredictor` trained on SageMaker into a local AutoGluon predictor with {py:meth}`~autogluon.cloud.TabularCloudPredictor.to_local_predictor`, as long as you have the same version of AutoGluon installed locally.

```python
local_predictor = cloud_predictor.to_local_predictor(
    save_path="PATH"  # If not specified, CloudPredictor will create one.
)  # local_predictor would be a TabularPredictor
```

`to_local_predictor()` downloads the trained model tarball, expands it to your local disk, and loads it as the corresponding AutoGluon predictor.
