

<div align="left">
  <img src="https://user-images.githubusercontent.com/16392542/77208906-224aa500-6aba-11ea-96bd-e81806074030.png" width="350">
</div>

# AutoGluon-Cloud

[![Continuous Integration](https://github.com/autogluon/autogluon-cloud/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/autogluon/autogluon-cloud/actions/workflows/continuous_integration.yml)

[AutoGluon-Cloud Documentation](https://auto.gluon.ai/cloud/stable/index.html) | [AutoGluon Documentation](https://auto.gluon.ai)

AutoGluon-Cloud aims to provide user tools to train, fine-tune and deploy [AutoGluon](https://auto.gluon.ai/stable/index.html) backed models on the cloud. With just a few lines of codes, users could train a model and perform inference on the cloud without worrying about MLOps details such as resource management.

Currently, AutoGluon-Cloud supports [AWS SageMaker](https://aws.amazon.com/sagemaker/) as the cloud backend.

## Installation
```bash
pip install -U pip
pip install -U setuptools wheel
pip install autogluon.cloud
```

## Example
```python

from autogluon.cloud import TabularCloudPredictor
import pandas as pd
train_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")
test_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")
test_data.drop(columns=['class'], inplace=True)
predictor_init_args = {"label": "class"}  # init args you would pass to AG TabularPredictor
predictor_fit_args = {"train_data": train_data, "time_limit": 120}  # fit args you would pass to AG TabularPredictor
cloud_predictor = TabularCloudPredictor(cloud_output_path='YOUR_S3_BUCKET_PATH')
cloud_predictor.fit(predictor_init_args=predictor_init_args, predictor_fit_args=predictor_fit_args)
cloud_predictor.deploy()
result = cloud_predictor.predict_real_time(test_data)
cloud_predictor.cleanup_deployment()
# Batch inference
result = cloud_predictor.predict(test_data)
```
