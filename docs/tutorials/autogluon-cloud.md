# Deploying AutoGluon Models with AutoGluon Cloud on AWS SageMaker
:label:`autogluon-cloud`

To help with AutoGluon models training, AWS developed a set of training and inference [deep learning containers](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#autogluon-training-containers). 
The containers can be used to train models with CPU and GPU instances and deployed as a SageMaker endpoint or used as a batch transform job.

We provide a full end-to-end example in [amazon-sagemaker-examples](https://github.com/aws/amazon-sagemaker-examples/tree/master/advanced_functionality/autogluon-tabular-containers) repository.

In addition, we now offer the (`autogluon.cloud`)[https://github.com/autogluon/autogluon-cloud] module to wrap functionality you saw in the example above with simple APIs.

## Pre-requisites
`autogluon.cloud` does not come with the default `autogluon` installation. Therefore, please install it via (`pip3 install autogluon.cloud`)
Also ensure that the latest version of sagemaker python API is installed via (`pip3 install --upgrade sagemaker`). 
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

## Training
Using `autogluon.cloud` to train AutoGluon backed models are simple and not too much different from training an AutoGluon predictor directly.

Currently, `autogluon.cloud` supports training/deploying `tabular`, `multimodal`, `text`, and `image` predictors. In the example below, we use `TabularCloudPredictor` for demonstration. You can substitute it with other `CloudPredictors` easily as they share the same APIs.

```{.python}
from autogluon.cloud import TabularCloudPredictor
train_data = 'train.csv'
predictor_init_args = {label='label'}  # init args you would pass to AG TabularPredictor
predictor_fit_args = {train_data, time_limit=120}  # fit args you would pass to AG TabularPredictor
cloud_predictor = TabularCloudPredictor(
    cloud_output_path='YOUR_S3_BUCKET_PATH'
).fit(
    predictor_init_args,
    predictor_fit_args,
    instance_type="INSTANCE_TYPE"  # Checkout supported instance and pricing here: https://aws.amazon.com/sagemaker/pricing/
)
```

To retrieve general info about the fitted `CloudPredictor`

```{.python}
cloud_predictor.info()
```

### Reattach to a Previous Training Job
If your local connection to the training job died for some reason, i.e. lost internet connection, your training job will still be going on SageMaker, and you can reattach to the job with another `CloudPredictor` as long as you have the job name.

```{.python}
another_cloud_predictor = TabularCloudPredictor(cloud_output_path='YOUR_S3_BUCKET_PATH')
another_cloud_predictor.attach_job(job_name="JOB_NAME")
```

To be noticed, the reattached job will no longer give live stream of the training job's log. Instead, the log will be available once the training job is finished.

## Endpoint Deployment and Real-time Prediction
If you want to deploy a predictor as a SageMaker endpoint, which can be used to do real-time inference later, it is just one line of code:

```{.python}
cloud_predictor.deploy(
    instance_type="INSTANCE_TYPE"  # Checkout supported instance and pricing here: https://aws.amazon.com/sagemaker/pricing/
)
```

You can also attach to a deployed endpoint,

```{.python}
cloud_predictor.attach_endpoint(endpoint="ENDPOINT_NAME")
```

To do real-time prediction,

```{.python}
result = cloud_predictor.predict_real_time(
    'test.csv',
    instance_type="INSTANCE_TYPE"  # Checkout supported instance and pricing here: https://aws.amazon.com/sagemaker/pricing/
)  # result will be a pandas dataframe
```

## Batch Inference
When minimizing latency isn't a concern, then the batch inference functionality may be easier, more scalable, and more appropriate.
To do a batch inference,

```{.python}
cloud_predictor.predict('test.csv')  # results will be stored in s3 bucket
cloud_predictor.download_predict_results()  # download the results to your local machine
```

## Convert the CloudPredictor to a Local AutoGluon Predictor
You can easily convert the `CloudPredictor` you trained on SageMaker to your local machine as long as you have the same version of AutoGluon installed locally.

```{.python}
local_predictor = cloud_predictor.to_local_predictor()  # local_predictor would be a TabularPredictor
```

## Note on Image Modality
If your training and inference tasks involve image modality, you need to
1. Make sure your column contains **absolute paths** to the images
2. provide argument `image_column` as the column name containing image paths to `CloudPredictor` fit/inference APIs.

```{.python}
cloud_predictor.fit(..., image_column="IMAGE_COLUMN_NAME")
cloud_predictor.predict_real_time(..., image_column="IMAGE_COLUMN_NAME")
cloud_predictor.predict(..., image_column="IMAGE_COLUMN_NAME")
```

## What Containers AutoGluon Cloud Support?
`autogluon.cloud` supports containers starting AutoGluon DLC 0.6.0.
