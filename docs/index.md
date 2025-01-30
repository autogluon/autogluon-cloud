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

AutoGluon-Cloud: Train and Deploy AutoGluon on the Cloud

:::
::::

::::::

AutoGluon-Cloud aims to provide user tools to train, fine-tune and deploy [AutoGluon](<https://auto.gluon.ai/stable/index.html>) backed models on the cloud. With just a few lines of code, users can train a model and perform inference on the cloud without worrying about MLOps details such as resource management.

Currently, AutoGluon-Cloud supports [Amazon SageMaker](<https://aws.amazon.com/sagemaker/>) as the cloud backend.

## {octicon}`rocket` Quick Examples

:::{dropdown} Tabular
:animate: fade-in-slide-down
:open:
:color: primary

```python
import pandas as pd
from autogluon.cloud import TabularCloudPredictor

train_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")
test_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")
test_data.drop(columns=["class"], inplace=True)
predictor_init_args = {
    "label": "class"
}  # args used when creating TabularPredictor()
predictor_fit_args = {
    "train_data": train_data,
    "time_limit": 120
}  # args passed to TabularPredictor.fit()
cloud_predictor = TabularCloudPredictor(cloud_output_path="YOUR_S3_BUCKET_PATH")
cloud_predictor.fit(
    predictor_init_args=predictor_init_args, predictor_fit_args=predictor_fit_args
)
cloud_predictor.deploy()
result = cloud_predictor.predict_real_time(test_data)
cloud_predictor.cleanup_deployment()
# Batch inference
result = cloud_predictor.predict(test_data)
```
:::


:::{dropdown} Multimodal
:animate: fade-in-slide-down
:color: primary

```python
import pandas as pd
from autogluon.cloud import MultiModalCloudPredictor

train_data = pd.read_parquet("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet")
test_data = pd.read_parquet("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet")
test_data.drop(columns=["label"], inplace=True)
predictor_init_args = {
    "label": "label"
}  # args used when creating MultiModalPredictor()
predictor_fit_args = {
    "train_data": train_data
}  # args passed to MultiModalPredictor.fit()
cloud_predictor = MultiModalCloudPredictor(cloud_output_path="YOUR_S3_BUCKET_PATH")
cloud_predictor.fit(
    predictor_init_args=predictor_init_args, predictor_fit_args=predictor_fit_args
)
cloud_predictor.deploy()
result = cloud_predictor.predict_real_time(test_data)
cloud_predictor.cleanup_deployment()
# Batch inference
result = cloud_predictor.predict(test_data)
```
:::


:::{dropdown} TimeSeries
:animate: fade-in-slide-down
:color: primary

```python
import pandas as pd
from autogluon.cloud import TimeSeriesCloudPredictor

data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_tiny/train.csv")

predictor_init_args = {
    "target": "target",
    "prediction_length" : 24,
}  # args used when creating TimeSeriesPredictor()
predictor_fit_args = {
    "train_data": data,
    "time_limit": 120,
}  # args passed to TimeSeriesPredictor.fit()
cloud_predictor = TimeSeriesCloudPredictor(cloud_output_path="YOUR_S3_BUCKET_PATH")
cloud_predictor.fit(
    predictor_init_args=predictor_init_args,
    predictor_fit_args=predictor_fit_args,
    id_column="item_id",
    timestamp_column="timestamp",
)
cloud_predictor.deploy()
result = cloud_predictor.predict_real_time(data)
cloud_predictor.cleanup_deployment()
# Batch inference
result = cloud_predictor.predict(data)
```
:::


## {octicon}`package` Installation

![](https://img.shields.io/pypi/pyversions/autogluon.cloud)
![](https://img.shields.io/pypi/v/autogluon.cloud.svg)
![](https://img.shields.io/pypi/dm/autogluon.cloud)

```bash
pip install -U pip
pip install -U setuptools wheel
pip install --pre autogluon.cloud  # You don't need to install autogluon itself locally
pip install -U sagemaker  # This is required to ensure the information about newly released containers is available.
```

```{toctree}
---
caption: Tutorials
maxdepth: 3
hidden:
---

Cloud <tutorials/index>
```

```{toctree}
---
caption: Resources
maxdepth: 1
hidden:
---

Versions <versions.rst>
```

```{toctree}
---
caption: API
maxdepth: 1
hidden:
---

TabularCloudPredictor <api/autogluon.cloud.TabularCloudPredictor>
MultiModalCloudPredictor <api/autogluon.cloud.MultiModalCloudPredictor>
TimeSeriesCloudPredictor <api/autogluon.cloud.TimeSeriesCloudPredictor>
```
