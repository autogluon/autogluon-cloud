

<div align="center">
<img src="https://user-images.githubusercontent.com/16392542/77208906-224aa500-6aba-11ea-96bd-e81806074030.png" width="350">

## Train and Deploy AutoGluon in the Cloud

[![PyPI](https://img.shields.io/pypi/v/autogluon.cloud.svg)](https://pypi.org/project/autogluon.cloud/)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/autogluon.cloud/)
[![GitHub license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Continuous Integration](https://github.com/autogluon/autogluon-cloud/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/autogluon/autogluon-cloud/actions/workflows/continuous_integration.yml)

[AutoGluon-Cloud Documentation](https://auto.gluon.ai/cloud/stable/index.html) | [AutoGluon Documentation](https://auto.gluon.ai)

</div>

[AutoGluon](https://auto.gluon.ai/stable/index.html) is an open-source AutoML library that trains state-of-the-art ML models on tabular, time-series, and multimodal data with just a few lines of code. AutoGluon-Cloud takes that same API and runs it on AWS — train models and serve predictions on [Amazon SageMaker](https://aws.amazon.com/sagemaker/) without managing infrastructure or setting up a heavyweight ML environment on your local machine.

It supports two workflows:

- **[Train your own predictor](https://auto.gluon.ai/cloud/stable/tutorials/predictor-tabular.html)** — the same `fit → deploy → predict` workflow as local AutoGluon, with all the heavy lifting offloaded to SageMaker.
- **[Run pretrained foundation models](https://auto.gluon.ai/cloud/stable/tutorials/foundation-model-timeseries.html)** — deploy state-of-the-art pretrained models like [Chronos-2](https://huggingface.co/amazon/chronos-2) for zero-shot inference, with no training required.

## 💾 Installation & setup

```bash
pip install autogluon.cloud
```

Then provision the IAM role and S3 bucket AutoGluon-Cloud needs on AWS:

```python
from autogluon.cloud import bootstrap

bootstrap()
```

See the [Setup tutorial](https://auto.gluon.ai/cloud/stable/tutorials/setup.html) for the full walkthrough, including how to register an existing role and bucket instead.

## ⚙️ Train your own model

```python
from autogluon.cloud import TabularCloudPredictor

# `train_data` and `test_data` can be a local path, S3 URL, or pandas DataFrame
train_data = "https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv"
test_data = "https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv"

# Train
cloud_predictor = TabularCloudPredictor()
cloud_predictor.fit(
    train_data=train_data,
    predictor_init_args={"label": "class"},  # passed to TabularPredictor()
    predictor_fit_args={"time_limit": 120},  # passed to TabularPredictor.fit()
)

# Real-time inference endpoint
cloud_predictor.deploy()
result = cloud_predictor.predict_real_time(test_data)
cloud_predictor.cleanup_deployment()

# Batch prediction
result = cloud_predictor.predict(test_data)
```

## 🚀 Run a pretrained foundation model

```python
from autogluon.cloud import TimeSeriesFoundationModel

# `data` can be a local path, S3 URL, or pandas DataFrame
data = "https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_tiny/train.csv"

model = TimeSeriesFoundationModel("chronos-2")

# Batch prediction — no training required
predictions = model.predict(
    data=data,
    target="target",
    prediction_length=24,
)

# Real-time inference endpoint
endpoint = model.deploy()
predictions = endpoint.predict(
    data=data,
    target="target",
    prediction_length=24,
)
endpoint.delete_endpoint()
```
