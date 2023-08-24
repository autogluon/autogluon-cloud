# AutoGluon Cloud FAQ

## Supported Docker Containers
`autogluon.cloud` supports AutoGluon Deep Learning Containers version 0.6.0 and newer.

## How to use Previous Versions of AutoGluon containers
By default, `autogluon.cloud` will fetch the latest version of AutoGluon DLC. However, you can supply `framework_version` to fit/inference APIs to access previous versions, i.e.
```python
cloud_predictor.fit(..., framework_version="0.6")
```
It is always recommended to use the latest version as it has more features and up-to-date security patches.


## How to Build a Cloud Compatible Custom container
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

You would likely need to grant ECR access permissions to this image to the IAM role interacting with cloud module.
