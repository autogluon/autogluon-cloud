---
sd_hide_title: true
hide-toc: true
---

# AutoGluon-Cloud

::::::{div} landing-title
:style: "padding: 0.1rem 0.5rem 0.6rem 0; background-image: linear-gradient(315deg, #438ff9 0%, #3977B9 74%); clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem)); -webkit-clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem));"

::::{grid}
:reverse:
:gutter: 2 3 3 3
:margin: 4 4 1 2

:::{grid-item}
:columns: 12 4 4 4

```{image} ./_static/autogluon-s.png
:width: 200px
:class: sd-m-auto sd-animate-grow50-rot20
```
:::

:::{grid-item}
:columns: 12 8 8 8
:child-align: justify
:class: sd-text-white sd-fs-3

Train and Deploy AutoGluon in the Cloud

:::
::::

::::::

AutoGluon-Cloud lets you train and deploy state-of-the-art ML models for classification, regression, and time series forecasting on [Amazon SageMaker](https://aws.amazon.com/sagemaker/). All it takes is a few lines of code; AutoGluon-Cloud handles the infrastructure, dependencies, and glue code for you.

## {octicon}`light-bulb` Why AutoGluon-Cloud?

- **Works like local [AutoGluon](https://auto.gluon.ai/stable/index.html).** Pass in DataFrames, get predictions back — as convenient as working locally, with the compute handled by AWS.
- **No boilerplate.** No training scripts, inference handlers, or serialization code to write and maintain.
- **Official AWS containers.** Everything runs in the [AutoGluon Deep Learning Containers](https://aws.github.io/deep-learning-containers/), maintained and security-patched by AWS.
- **Sensible defaults, fully configurable.** Under the hood it's just SageMaker running in your AWS account, so you stay in full control.

## {octicon}`package` Installation

```bash
pip install autogluon.cloud
autogluon-cloud bootstrap  # one-time setup for IAM role and S3 bucket
```

See the [Setup tutorial](tutorials/setup.md) for more details.

## {octicon}`rocket` Foundation models

:::{dropdown} Time Series (Chronos-2)
:animate: fade-in-slide-down
:open:
:color: primary

Zero-shot forecasts with a pretrained model — no training required.

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

→ [Full walkthrough](tutorials/foundation-model-timeseries.md)
:::


## {octicon}`gear` Train your own predictor

:::{dropdown} Tabular
:animate: fade-in-slide-down
:color: primary

Train a classification or regression model on tabular data.

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

→ [Full walkthrough](tutorials/predictor-tabular.md)
:::


:::{dropdown} Time Series
:animate: fade-in-slide-down
:color: primary

Forecast future values of time series.

```python
from autogluon.cloud import TimeSeriesCloudPredictor

# `data` can be a local path, S3 URL, or pandas DataFrame
data = "https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_tiny/train.csv"

# Train
cloud_predictor = TimeSeriesCloudPredictor()
cloud_predictor.fit(
    train_data=data,
    predictor_init_args={"target": "target", "prediction_length": 24},  # passed to TimeSeriesPredictor()
    predictor_fit_args={"time_limit": 120},  # passed to TimeSeriesPredictor.fit()
)

# Batch prediction
result = cloud_predictor.predict(data)

# Real-time inference endpoint
endpoint = cloud_predictor.deploy()
result = endpoint.predict(data)
endpoint.delete_endpoint()
```

→ [Full walkthrough](tutorials/predictor-timeseries.md)
:::


```{toctree}
---
caption: Tutorials
maxdepth: 2
hidden:
---

Setup <tutorials/setup>
Train Your Own Predictor <tutorials/predictor>
Foundation Models <tutorials/foundation-model>
```

```{toctree}
---
caption: API
maxdepth: 1
hidden:
---

Setup <api/setup>
Tabular <api/tabular>
Time Series <api/timeseries>
Multimodal <api/multimodal>
```

```{toctree}
---
caption: Resources
maxdepth: 1
hidden:
---

Versions <versions.rst>
AutoGluon documentation <https://auto.gluon.ai/stable/index.html>
GitHub <https://github.com/autogluon/autogluon-cloud>
```
