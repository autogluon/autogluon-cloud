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

AutoGluon-Cloud makes it easy to run [AutoGluon](<https://auto.gluon.ai/stable/index.html>) in the cloud. With a few lines of code, you can train models and run inference on [Amazon SageMaker](<https://aws.amazon.com/sagemaker/>) — without managing infrastructure or installing AutoGluon's heavy dependencies on your local machine.

It supports two workflows:

- **Train AutoGluon predictors in the cloud** — the same `fit → deploy → predict` workflow as local AutoGluon, with all the heavy lifting offloaded to SageMaker.
- **Run pretrained foundation models** — deploy state-of-the-art pretrained models like Chronos-2 for zero-shot inference, with no training required.

## {octicon}`gear` Train AutoGluon predictors in the cloud

*Full walkthrough: [Train Your Own Models](tutorials/cloud-predictor.md)*

:::{dropdown} Tabular
:animate: fade-in-slide-down
:open:
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

# Real-time inference endpoint
cloud_predictor.deploy()
result = cloud_predictor.predict_real_time(test_data)
cloud_predictor.cleanup_deployment()

# Batch prediction
result = cloud_predictor.predict(test_data)
```
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

# Real-time inference endpoint
cloud_predictor.deploy()
result = cloud_predictor.predict_real_time(data)
cloud_predictor.cleanup_deployment()

# Batch prediction
result = cloud_predictor.predict(data)
```
:::


## {octicon}`rocket` Run pretrained foundation models

*Full walkthrough: [Use Foundation Models](tutorials/foundation-model.md)*

:::{dropdown} Time Series (Chronos-2)
:animate: fade-in-slide-down
:color: primary

Zero-shot forecasts with a pretrained model — no training required.

```python
from autogluon.cloud import TimeSeriesFoundationModel

# `data` can be a local path, S3 URL, or pandas DataFrame
data = "https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_tiny/train.csv"

model = TimeSeriesFoundationModel("chronos-2")

# Batch prediction
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
:::


## {octicon}`package` Installation

![](https://img.shields.io/pypi/pyversions/autogluon.cloud)
![](https://img.shields.io/pypi/v/autogluon.cloud.svg)
![](https://img.shields.io/pypi/dm/autogluon.cloud)

```bash
pip install autogluon.cloud
```

Before running the examples above, set up your AWS resources (IAM role + S3 bucket) by following the [Setup](tutorials/setup.md) tutorial.

```{toctree}
---
caption: Tutorials
maxdepth: 1
hidden:
---

Setup <tutorials/setup>
Train Your Own Models <tutorials/cloud-predictor>
Use Foundation Models <tutorials/foundation_model>
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
