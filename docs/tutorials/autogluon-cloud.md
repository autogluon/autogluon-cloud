# Deploying AutoGluon Models with AutoGluon Cloud on AWS SageMaker
:label:`autogluon-cloud`

To help with AutoGluon models training, AWS developed a set of training and inference [deep learning containers](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#autogluon-training-containers). 
The containers can be used to train models with CPU and GPU instances and deployed as a SageMaker endpoint or used as a batch transform job.

We offer the [autogluon.cloud](https://github.com/autogluon/autogluon-cloud) module to utilize those containers and [AWS SageMaker](https://aws.amazon.com/sagemaker/) underneath to train/deploy AutoGluon backed models with simple APIs.

**Costs for running cloud compute are managed by AWS SageMaker, and storage costs are managed by AWS S3. AutoGluon-Cloud is a wrapper to these services at no additional charge. While AutoGluon-Cloud makes an effort to simplify the usage of these services, it is ultimately the user's responsibility to monitor compute usage within their account to ensure no unexpected charges.**

## Installation
`autogluon.cloud` does not come with the default `autogluon` installation. You can install it via:

```{.bash}
pip3 install autogluon.cloud
```

Also ensure that the latest version of sagemaker python API is installed via:

```{.bash}
pip3 install --upgrade sagemaker
```

This is required to ensure the information about newly released containers is available.

## Prepare an AWS Role with Necessary Permissions
`autogluon.cloud` utilizes various AWS resources to operate.
To help you to setup the necessary permissions, you can generate trust relationship and iam policy with our utils through

```{.python}
from autogluon.cloud import TabularCloudPredictor  # Can be other CloudPredictor as well

TabularCloudPredictor.generate_trust_relationship_and_iam_policy_file(
    account_id="YOUR_ACCOUNT_ID",  # The AWS account ID you plan to use for CloudPredictor.
    cloud_output_bucket="S3_BUCKET"  # S3 bucket name where intermediate artifacts will be uploaded and trained models should be saved. You need to create this bucket beforehand.
)
```

The util function above would give you two json files describing the trust replationship and the iam policy.
**Make sure you review those files and make necessary changes according to your use case before applying them.**

We recommend you to create an IAM Role for your IAM User to delegate as IAM Role doesn't have permanent long-term credentials and is used to directly interact with AWS services.
Refer to this [tutorial](https://aws.amazon.com/premiumsupport/knowledge-center/iam-assume-role-cli/) to

1. create the IAM Role with the trust relationship and iam policy you generated above
2. setup the credential
3. assume the role

## Training
Using `autogluon.cloud` to train AutoGluon backed models is simple and not too much different from training an AutoGluon predictor directly.

Currently, `autogluon.cloud` supports training/deploying `tabular`, `multimodal`, `text`, and `image` predictors. In the example below, we use `TabularCloudPredictor` for demonstration. You can substitute it with other `CloudPredictors` easily as they share the same APIs.

```{.python}
from autogluon.cloud import TabularCloudPredictor
train_data = 'train.csv'  # can be a DataFrame as well
predictor_init_args = {label='label'}  # init args you would pass to AG TabularPredictor
predictor_fit_args = {train_data, time_limit=120}  # fit args you would pass to AG TabularPredictor
cloud_predictor = TabularCloudPredictor(
    cloud_output_path='YOUR_S3_BUCKET_PATH'
).fit(
    predictor_init_args,
    predictor_fit_args,
    instance_type="ml.m5.2xlarge"  # Checkout supported instance and pricing here: https://aws.amazon.com/sagemaker/pricing/
    wait=True  # Set this to False to make it an unblocking call and immediately return
)
```

### Reattach to a Previous Training Job
If your local connection to the training job died for some reason, i.e. lost internet connection, your training job will still be running on SageMaker, and you can reattach to the job with another `CloudPredictor` as long as you have the job name.

The job name will be logged out when the training job started.
It should look similar to this: `INFO:sagemaker:Creating training-job with name: ag-cloudpredictor-1673296750-47d7`.
Alternatively, you can go to the SageMaker console and find the ongoing training job and its corresponding job name.

```{.python}
another_cloud_predictor = TabularCloudPredictor(cloud_output_path='YOUR_S3_BUCKET_PATH')
another_cloud_predictor.attach_job(job_name="JOB_NAME")
```

The reattached job will no longer give live stream of the training job's log. Instead, the log will be available once the training job is finished.

## Endpoint Deployment and Real-time Prediction
If you want to deploy a predictor as a SageMaker endpoint, which can be used to do real-time inference later, it is just one line of code:

```{.python}
cloud_predictor.deploy(
    instance_type="ml.m5.2xlarge",  # Checkout supported instance and pricing here: https://aws.amazon.com/sagemaker/pricing/
    wait=True  # Set this to False to make it an unblocking call and immediately return
)
```

Optionally, you can also attach to a deployed endpoint:

```{.python}
cloud_predictor.attach_endpoint(endpoint="ENDPOINT_NAME")
```

To perform real-time prediction:

```{.python}
result = cloud_predictor.predict_real_time('test.csv') # can be a DataFrame as well
```

Result would be a pandas Series similar to this:

```{.python}
0      dog
1      cat
2      cat
Name: label, dtype: object
```

To perform real-time predict probability:

```{.python}
result = cloud_predictor.predict_proba_real_time('test.csv')  # can be a DataFrame as well
```

Result would be a pandas DataFrame similar to this:

```{.python}
         dog       cat
0   0.682754  0.317246
1   0.195782  0.804218
2   0.372283  0.627717
```

Make sure you clean up the endpoint deployed by:

```{.python}
cloud_predictor.cleanup_deployment()
```

To identify if you have an active endpoint attached:

```{.python}
cloud_predictor.info()
```

The code above would return you a dict showing general info of the CloudPredictor.
One key inside would be `endpoint`, and it will tell you the name of the endpoint if there's an attached one, i.e.

```{.python}
{
    ...
    'endpoint': 'ag-cloudpredictor-1668189208-d23b'
}
```

## Batch Inference
When minimizing latency isn't a concern, then the batch inference functionality may be easier, more scalable, and cheaper as compute is automatically terminated after the batch inference job is complete.

A general guideline is to use batch inference if you need to get predictions less than once an hour and are ok with the inference time taking 10 minutes longer than real-time inference (due to compute spin-up overhead).

To perform batch inference:

```{.python}
result = cloud_predictor.predict(
    'test.csv',  # can be a DataFrame as well and the results will be stored in s3 bucket
    instance_type="ml.m5.2xlarge",  # Checkout supported instance and pricing here: https://aws.amazon.com/sagemaker/pricing/
    wait=True,  # Set this to False to make it an unblocking call and immediately return
    # If True, returns a Pandas Series object of predictions.
    # If False, returns nothing. You will have to download results separately via cloud_predictor.download_predict_results
    download=True,
    persist=True,  # If True and download=True, the results file will also be saved to local disk.
    save_path=None  # Path to save the downloaded results. If None, CloudPredictor will create one with the batch inference job name.
)
```

Result would be a pandas DataFrame similar to this:

```{.python}
   pred   0_proba   1_proba
0     1  0.317246  0.682754
1     1  0.195782  0.804218
```

## Retrieve CloudPredictor Info

To retrieve general info about a `CloudPredictor`

```{.python}
cloud_predictor.info()
```

It will output a dict similar to this:

```{.python}
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

```{.python}
local_predictor = cloud_predictor.to_local_predictor(
    save_path="PATH"  # If not specified, CloudPredictor will create one.
)  # local_predictor would be a TabularPredictor
```

`to_local_predictor()` would underneath downlod the tarball, expand it to your local disk and load it as a corresponding AutoGluon predictor.

## Training/Inference with Image Modality
If your training and inference tasks involve image modality, your data would contain a column representing the path to the image file, i.e.

```{.python}
   feature_1                     image   label
0          1   image/train/train_1.png       0
1          2   image/train/train_1.png       1
```

### Preparing the Image Column
Currently, AutoGluon only supports one image per row.
If your dataset contains one or more images per row, we first need to preprocess the image column to only contain the first image of each row.

For example, if your images are seperated with `;`, you can preprocess it via:

```{.python}
# image_col is the column name containing the image path. In the example above, it would be `image`
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])
```

Now we update the path to an absolute path.

For example, if your directory is similar to this:

```{.bash}
.
└── current_working_directory/
    ├── train.csv
    ├── test.csv
    └── images/
        ├── train/
        │   └── train_1.png
        └── test/
            └── test_1.png
```

You can replace your image column to absolute paths via:

```{.python}
train_data[image_col] = train_data[image_col].apply(lambda path: os.path.abspath(path))
test_data[image_col] = test_data[image_col].apply(lambda path: os.path.abspath(path))
```

### Perform Training/Inference with Image Modality
Provide argument `image_column` as the column name containing image paths to `CloudPredictor` fit/inference APIs.
In the example above, `image_column` would be `image`

```{.python}
cloud_predictor.fit(..., image_column="IMAGE_COLUMN_NAME")
cloud_predictor.predict_real_time(..., image_column="IMAGE_COLUMN_NAME")
cloud_predictor.predict(..., image_column="IMAGE_COLUMN_NAME")
```

## Supported Docker Containers
`autogluon.cloud` supports AutoGluon Deep Learning Containers version 0.6.0 and newer.

### Use Custom Containers
Though not recommended, `autogluon.cloud` supports using your custom containers by specifying `custom_image_uri`.

```{.python}
cloud_predictor.fit(..., custom_image_uri="CUSTOM_IMAGE_URI")
cloud_predictor.predict_real_time(..., custom_image_uri="CUSTOM_IMAGE_URI")
cloud_predictor.predict(..., custom_image_uri="CUSTOM_IMAGE_URI")
```
