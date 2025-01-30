# AutoGluon-Cloud FAQ

## Supported Docker Containers
`autogluon.cloud` supports AutoGluon Deep Learning Containers version 0.6.0 and newer.

## How to use Previous Versions of AutoGluon containers
By default, `autogluon.cloud` will fetch the latest version of AutoGluon DLC. However, you can supply `framework_version` to fit/inference APIs to access previous versions, i.e.
```python
cloud_predictor.fit(..., framework_version="0.6")
```
It is always recommended to use the latest version as it has more features and up-to-date security patches.


## How to Build a Cloud Compatible Custom Container
If the official DLC doesn't meet your requirement, and you would like to build your own container.

You can either build on top of our [DLC](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#autogluon-training-containers)
or refer to our [Dockerfiles](https://github.com/aws/deep-learning-containers/tree/master/autogluon)

## How to Use Custom Containers
Though not recommended, `autogluon.cloud` supports using your custom containers by specifying `custom_image_uri`.

```python
cloud_predictor.fit(..., custom_image_uri="CUSTOM_IMAGE_URI")
cloud_predictor.predict_real_time(..., custom_image_uri="CUSTOM_IMAGE_URI")
cloud_predictor.predict(..., custom_image_uri="CUSTOM_IMAGE_URI")
```

If this custom image lives under a certain ECR, you would need to grant access permission to the IAM role used by the Cloud module.

## Run into Permission Issues
You can try to get the necessary IAM permission and trust relationship through
```python
from autogluon.cloud import TabularCloudPredictor  # Can be other CloudPredictor as well

TabularCloudPredictor.generate_default_permission(
    backend="BACKNED_YOU_WANT"  # We currently support sagemaker and ray_aws
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
