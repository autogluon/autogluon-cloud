# Train and Deploy AutoGluon Models with AutoGluon-Cloud

AutoGluon-Cloud lets you train, deploy, and run inference with AutoGluon models on AWS using the same APIs you'd use locally. Under the hood, it runs your jobs on [Amazon SageMaker](https://aws.amazon.com/sagemaker/) using AWS's official [AutoGluon deep learning containers](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#autogluon-training-containers) — so you don't manage any infrastructure yourself.

It supports the `tabular`, `timeseries`, and `multimodal` predictors. The examples below use `TabularCloudPredictor`; the others share the same API.

```{note}
This tutorial assumes you've already set up AutoGluon-Cloud on AWS. If you haven't, see [Setup](setup.md) first.
```

```{attention}
SageMaker compute and S3 storage are billed to your AWS account. AutoGluon-Cloud is a free wrapper, but it's your responsibility to monitor usage to avoid unexpected charges.
```

## Training
Using `autogluon.cloud` to train AutoGluon backed models is simple and not too much different from training an AutoGluon predictor directly.

```python
from autogluon.cloud import TabularCloudPredictor
cloud_predictor = TabularCloudPredictor(
    cloud_output_path="YOUR_S3_BUCKET_PATH"
).fit(
    train_data="train.csv",  # path or DataFrame
    predictor_init_args={"label": "label"},  # passed to TabularPredictor()
    predictor_fit_args={"time_limit": 120},  # passed to TabularPredictor.fit()
    instance_type="ml.m5.2xlarge",  # https://aws.amazon.com/sagemaker/pricing/
    wait=True,  # Set this to False for an unblocking call
)
```

### Reattach to a Previous Training Job
If your local connection to the training job died for some reason, i.e. lost internet connection, your training job will still be running on SageMaker, and you can reattach to the job with another `CloudPredictor` as long as you have the job name.

The job name will be logged out when the training job started.
It should look similar to this: `INFO:sagemaker:Creating training-job with name: ag-cloudpredictor-1673296750-47d7`.
Alternatively, you can go to the SageMaker console and find the ongoing training job and its corresponding job name.

```python
another_cloud_predictor = TabularCloudPredictor()
another_cloud_predictor.attach_job(job_name="JOB_NAME")
```

The reattached job will no longer give live stream of the training job's log. Instead, the log will be available once the training job is finished.

## Endpoint Deployment and Real-time Prediction
If you want to deploy a predictor as a SageMaker endpoint, which can be used to do real-time inference later, it is just one line of code:

```python
cloud_predictor.deploy(
    instance_type="ml.m5.2xlarge",  # Checkout supported instance and pricing here: https://aws.amazon.com/sagemaker/pricing/
    wait=True,  # Set this to False to make it an unblocking call and immediately return
)
```

Optionally, you can also attach to a deployed endpoint:

```python
cloud_predictor.attach_endpoint(endpoint="ENDPOINT_NAME")
```

To perform real-time prediction:

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

To perform real-time predict probability:

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

Make sure you clean up the endpoint deployed by:

```python
cloud_predictor.cleanup_deployment()
```

To identify if you have an active endpoint attached:

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

## Batch Inference
When minimizing latency isn't a concern, then the batch inference functionality may be easier, more scalable, and cheaper as compute is automatically terminated after the batch inference job is complete.

A general guideline is to use batch inference if you need to get predictions less than once an hour and are ok with the inference time taking 10 minutes longer than real-time inference (due to compute spin-up overhead).

To perform batch inference:

```python
result = cloud_predictor.predict(
    'test.csv',  # can be a DataFrame as well and the results will be stored in s3 bucket
    instance_type="ml.m5.2xlarge",  # Checkout supported instance and pricing here: https://aws.amazon.com/sagemaker/pricing/
    wait=True,  # Set this to False to make it an unblocking call and immediately return
    # If True, returns a Pandas Series object of predictions.
    # If False, returns nothing. You will have to download results separately via cloud_predictor.download_predict_results
    download=True,
    persist=True,  # If True and download=True, the results file will also be saved to local disk.
    save_path=None,  # Path to save the downloaded results. If None, CloudPredictor will create one with the batch inference job name.
)
```

Result would be a pandas DataFrame similar to this:

```python
0      dog
1      cat
2      cat
Name: label, dtype: object
```

To perform batch inference and getting prediction probability:

```python
result = cloud_predictor.predict_proba(
    'test.csv',  # can be a DataFrame as well and the results will be stored in s3 bucket
    include_predict=True,  # Will return a tuple (prediction, prediction probability). Set this to False to get prediction probability only.
    instance_type="ml.m5.2xlarge",  # Checkout supported instance and pricing here: https://aws.amazon.com/sagemaker/pricing/
    wait=True,  # Set this to False to make it an unblocking call and immediately return
    # If True, returns a Pandas Series object of predictions.
    # If False, returns nothing. You will have to download results separately via cloud_predictor.download_predict_results
    download=True,
    persist=True,  # If True and download=True, the results file will also be saved to local disk.
    save_path=None,  # Path to save the downloaded results. If None, CloudPredictor will create one with the batch inference job name.
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

To retrieve general info about a `CloudPredictor`

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
You can easily convert the `CloudPredictor` you trained on SageMaker to your local machine as long as you have the same version of AutoGluon installed locally.

```python
local_predictor = cloud_predictor.to_local_predictor(
    save_path="PATH"  # If not specified, CloudPredictor will create one.
)  # local_predictor would be a TabularPredictor
```

`to_local_predictor()` would underneath downlod the tarball, expand it to your local disk and load it as a corresponding AutoGluon predictor.
