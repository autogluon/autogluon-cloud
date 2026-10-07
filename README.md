

<div align="center">
<img src="https://user-images.githubusercontent.com/16392542/77208906-224aa500-6aba-11ea-96bd-e81806074030.png" width="350">

## Train and Deploy AutoGluon in the Cloud

[![PyPI](https://img.shields.io/pypi/v/autogluon.cloud.svg)](https://pypi.org/project/autogluon.cloud/)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/autogluon.cloud/)
[![GitHub license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Continuous Integration](https://github.com/autogluon/autogluon-cloud/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/autogluon/autogluon-cloud/actions/workflows/continuous_integration.yml)

[AutoGluon-Cloud Documentation](https://auto.gluon.ai/cloud/stable/index.html) | [AutoGluon Documentation](https://auto.gluon.ai)

</div>

AutoGluon-Cloud lets you train and deploy state-of-the-art ML models for classification, regression, and time series forecasting on [Amazon SageMaker](https://aws.amazon.com/sagemaker/). All it takes is a few lines of code; AutoGluon-Cloud handles the infrastructure, dependencies, and glue code for you.

## 💡 Why AutoGluon-Cloud?

- **Works like local [AutoGluon](https://auto.gluon.ai/stable/index.html).** Pass in DataFrames, get predictions back — as convenient as working locally, with the compute handled by AWS.
- **No boilerplate.** No training scripts, inference handlers, or serialization code to write and maintain.
- **Official AWS containers.** Everything runs in the [AutoGluon Deep Learning Containers](https://aws.github.io/deep-learning-containers/), maintained and security-patched by AWS.
- **Sensible defaults, fully configurable.** Under the hood it's just SageMaker running in your AWS account, so you stay in full control.

## 💾 Installation

```bash
pip install autogluon.cloud
autogluon-cloud bootstrap  # one-time setup for IAM role and S3 bucket
```

See the [Setup tutorial](https://auto.gluon.ai/cloud/stable/tutorials/setup.html) for more details.

## 🚀 Foundation models

Zero-shot forecasts with a pretrained model like [Chronos-2](https://huggingface.co/amazon/chronos-2) — no training required. [Full walkthrough](https://auto.gluon.ai/cloud/stable/tutorials/foundation-model-timeseries.html).

```python
from autogluon.cloud import TimeSeriesFoundationModel

# `data` can be a local path, S3 URL, or pandas DataFrame
data = "https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_tiny/train.csv"

model = TimeSeriesFoundationModel("chronos-2")

# Batch prediction
predictions = model.predict(data=data, target="target", prediction_length=24)

# Real-time inference endpoint
endpoint = model.deploy()
predictions = endpoint.predict(data=data, target="target", prediction_length=24)
endpoint.delete_endpoint()
```

## ⚙️ Train your own predictor

Train an AutoGluon predictor on your data and serve it from SageMaker. Full walkthrough: [time series](https://auto.gluon.ai/cloud/stable/tutorials/predictor-timeseries.html), [tabular](https://auto.gluon.ai/cloud/stable/tutorials/predictor-tabular.html).

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

# Batch prediction
result = cloud_predictor.predict(test_data)

# Real-time inference endpoint
endpoint = cloud_predictor.deploy()
result = endpoint.predict(test_data)
endpoint.delete_endpoint()
```
