# Train and Deploy AutoGluon Models with AutoGluon-Cloud

AutoGluon-Cloud lets you train, deploy, and run inference with AutoGluon models on AWS using the same APIs you'd use locally. Under the hood, it runs your jobs on [Amazon SageMaker](https://aws.amazon.com/sagemaker/) using AWS's official [AutoGluon deep learning containers](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#autogluon-training-containers) — so you don't manage any infrastructure yourself.

It supports the `tabular`, `timeseries`, and `multimodal` predictors. The examples below use {py:class}`~autogluon.cloud.TabularCloudPredictor`; the others share the same API.

```{note}
This tutorial assumes you've already set up AutoGluon-Cloud on AWS. If you haven't, see [Setup](setup.md) first.
```

```{attention}
SageMaker compute and S3 storage are billed to your AWS account. AutoGluon-Cloud is a free wrapper, but it's your responsibility to monitor usage to avoid unexpected charges.
```

## Training

**Create the predictor.** A {py:class}`~autogluon.cloud.TabularCloudPredictor` needs an IAM execution role (so SageMaker can run jobs on your behalf) and an S3 bucket (to stage data and store trained artifacts). There are two ways to supply them:

- Use a saved config (recommended). Save the role and bucket once to `~/.autogluon/cloud.yaml` — see [Setup](setup.md) — and subsequent constructor calls will pick them up automatically:

  ```python
  from autogluon.cloud import TabularCloudPredictor

  cloud_predictor = TabularCloudPredictor()
  ```

- Pass them at construction. Useful when you need different roles or buckets per call:

  ```python
  cloud_predictor = TabularCloudPredictor(
      role="arn:aws:iam::222222222222:role/MyAutoGluonRole",
      cloud_output_path="s3://my-autogluon-bucket/tabular-demo",
  )
  ```

**Train.** {py:meth}`~autogluon.cloud.TabularCloudPredictor.fit` runs [`TabularPredictor.fit()`](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html) inside a remote SageMaker job — along with `train_data`, the `predictor_init_args` and `predictor_fit_args` are forwarded straight through. Training, model artifacts, and AutoGluon itself all live on the remote instance, so you don't need AutoGluon installed locally.

`train_data` can be a pandas DataFrame, or a path to a local or S3 file (CSV or Parquet). In every case AutoGluon-Cloud loads the data locally and uploads it to your `cloud_output_path` bucket before kicking off the SageMaker job.

```python
cloud_predictor.fit(
    train_data="train.csv",  # DataFrame, local path, or S3 URL (CSV/Parquet)
    predictor_init_args={"label": "label"},  # passed to TabularPredictor()
    predictor_fit_args={"time_limit": 120},  # passed to TabularPredictor.fit()
    instance_type="ml.m5.2xlarge",  # https://aws.amazon.com/sagemaker/pricing/
    wait=True,  # Set this to False for an unblocking call
)
```

### Reattach to a Previous Training Job
If your local connection to the training job died for some reason, i.e. lost internet connection, your training job will still be running on SageMaker, and you can reattach to the job with another `CloudPredictor` via {py:meth}`~autogluon.cloud.TabularCloudPredictor.attach_job` as long as you have the job name.

The job name will be logged out when the training job started.
It should look similar to this: `INFO:sagemaker:Creating training-job with name: ag-cloudpredictor-1673296750-47d7`.
Alternatively, you can go to the SageMaker console and find the ongoing training job and its corresponding job name.

```python
another_cloud_predictor = TabularCloudPredictor()
another_cloud_predictor.attach_job(job_name="JOB_NAME")
```

The reattached job will no longer give live stream of the training job's log. Instead, the log will be available once the training job is finished.

## Inference

Once a predictor is trained, you can get predictions in two ways:

- **Real-time inference**: deploy the predictor as a long-running SageMaker endpoint and send requests to it. Best when you need low-latency predictions on demand — e.g. behind a user-facing service.
- **Batch inference**: launch a one-off SageMaker job that scores a dataset and writes the results to S3. Best for offline scoring of larger datasets — compute spins up, runs, and shuts down automatically, so you only pay for what you use.

A rough guideline: if you need predictions less often than once an hour and can tolerate ~10 minutes of compute spin-up, batch inference is usually cheaper and easier to operate.

### Real-time inference

Deploy the predictor as a SageMaker endpoint with {py:meth}`~autogluon.cloud.TabularCloudPredictor.deploy`:

```python
cloud_predictor.deploy(
    instance_type="ml.m5.2xlarge",  # Checkout supported instance and pricing here: https://aws.amazon.com/sagemaker/pricing/
    wait=True,  # Set this to False to make it an unblocking call and immediately return
)
```

Optionally, you can also attach to a deployed endpoint with {py:meth}`~autogluon.cloud.TabularCloudPredictor.attach_endpoint`:

```python
cloud_predictor.attach_endpoint(endpoint="ENDPOINT_NAME")
```

Send requests to the endpoint with {py:meth}`~autogluon.cloud.TabularCloudPredictor.predict_real_time`:

```python
result = cloud_predictor.predict_real_time("test.csv") # can be a DataFrame as well
```

Result would be a pandas Series similar to this:

```python
0      dog
1      cat
2      cat
Name: label, dtype: object
```

For class probabilities, use {py:meth}`~autogluon.cloud.TabularCloudPredictor.predict_proba_real_time`:

```python
result = cloud_predictor.predict_proba_real_time("test.csv")  # can be a DataFrame as well
```

Result would be a pandas DataFrame similar to this:

```python
         dog       cat
0   0.682754  0.317246
1   0.195782  0.804218
2   0.372283  0.627717
```

Make sure you clean up the endpoint with {py:meth}`~autogluon.cloud.TabularCloudPredictor.cleanup_deployment`:

```python
cloud_predictor.cleanup_deployment()
```

To check whether an endpoint is attached, call {py:meth}`~autogluon.cloud.TabularCloudPredictor.info`:

```python
cloud_predictor.info()
```

The code above would return you a dict showing general info of the CloudPredictor.
One key inside would be `endpoint`, and it will tell you the name of the endpoint if there's an attached one, i.e.

```python
{
    ...
    'endpoint': 'ag-cloudpredictor-1668189208-d23b'
}
```

### Invoke the Endpoint without AutoGluon-Cloud
The endpoint being deployed is a normal Sagemaker Endpoint, and you can invoke it through other methods. For example, to invoke an endpoint with boto3 directly

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

To score a dataset as a one-off job, use {py:meth}`~autogluon.cloud.TabularCloudPredictor.predict`:

```python
result = cloud_predictor.predict(
    "test.csv",  # DataFrame, local path, or S3 URL (CSV/Parquet)
    instance_type="ml.m5.2xlarge",
)
```

Result would be a pandas DataFrame similar to this:

```python
0      dog
1      cat
2      cat
Name: label, dtype: object
```

For class probabilities, use {py:meth}`~autogluon.cloud.TabularCloudPredictor.predict_proba`:

```python
result = cloud_predictor.predict_proba(
    "test.csv",
    instance_type="ml.m5.2xlarge",
)
```

Result would be a tuple containing both the prediction and prediction probability if `include_predict` is True, i.e.

```python
0      dog
1      cat
2      cat
Name: label, dtype: object
,
         dog       cat
0   0.682754  0.317246
1   0.195782  0.804218
2   0.372283  0.627717
```

Otherwise, prediction probability only, i.e.

```python
         dog       cat
0   0.682754  0.317246
1   0.195782  0.804218
2   0.372283  0.627717
```

## Retrieve CloudPredictor Info

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

## Convert the CloudPredictor to a Local AutoGluon Predictor
You can convert the `CloudPredictor` trained on SageMaker into a local AutoGluon predictor with {py:meth}`~autogluon.cloud.TabularCloudPredictor.to_local_predictor`, as long as you have the same version of AutoGluon installed locally.

```python
local_predictor = cloud_predictor.to_local_predictor(
    save_path="PATH"  # If not specified, CloudPredictor will create one.
)  # local_predictor would be a TabularPredictor
```

`to_local_predictor()` downloads the trained model tarball, expands it to your local disk, and loads it as the corresponding AutoGluon predictor.
