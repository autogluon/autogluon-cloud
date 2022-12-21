

<div align="left">
  <img src="https://user-images.githubusercontent.com/16392542/77208906-224aa500-6aba-11ea-96bd-e81806074030.png" width="350">
</div>

# AutoGluon-Cloud

[![Continuous Integration](https://github.com/autogluon/autogluon-cloud/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/autogluon/autogluon-cloud/actions/workflows/continuous_integration.yml)

AutoGluon-Cloud aims to provide user tools to train, fine-tune and deploy [AutoGluon](https://auto.gluon.ai/stable/index.html) backed models on the cloud. With just a few lines of codes, users could train a model and perform inference on the cloud without worrying about MLOps details such as resource management.

Currently, AutoGluon-Cloud supports [AWS SageMaker](https://aws.amazon.com/sagemaker/) as the cloud backend.

## Example
```python
# First install package from terminal:
# pip install -U pip
# pip install -U setuptools wheel
# pip install --pre autogluon.cloud  # You don't need to install autogluon itself locally

from autogluon.cloud import TabularCloudPredictor
train_data = 'train.csv'
test_data = 'test.csv'
predictor_init_args = {label='label'}  # init args you would pass to AG TabularPredictor
predictor_fit_args = {train_data, time_limit=120}  # fit args you would pass to AG TabularPredictor
# Train
cloud_predictor = TabularCloudPredictor(cloud_output_path='YOUR_S3_BUCKET_PATH').fit(predictor_init_args, predictor_fit_args)
# Deploy the endpoint
cloud_predictor.deploy()
# Real-time inference with the endpoint
result = cloud_predictor.predict_real_time('test.csv')
print(result)
# Cleanup the endpoint
cloud_predictor.cleanup_deployment()
# Batch inference
cloud_predictor.predict('test.csv')  # results will be stored in s3 bucket
cloud_predictor.download_predict_results()  # download the results to your local machine
```
