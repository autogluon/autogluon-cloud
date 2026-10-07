# AutoGluon Container

Every SageMaker job and endpoint launched by AutoGluon-Cloud runs in the official AutoGluon [Deep Learning Container](https://aws.github.io/deep-learning-containers/) (DLC). The image is built and maintained by AWS, security-patched on an Amazon Linux 2023 base, and includes AutoGluon together with all the libraries it builds on.

You can also use the image directly — with your own training scripts and inference handlers — when you need more control than AutoGluon-Cloud provides. This section documents the image for both cases.

::::{grid} 2
  :gutter: 3

:::{grid-item-card} Custom Scripts
  :link: custom-scripts.html

  Train and serve your own AutoGluon scripts on SageMaker with the image.
:::

:::{grid-item-card} Reference
  :link: reference.html

  Inference handler contract, environment variables, and known limitations.
:::

::::

## Images

A single image runs both SageMaker training jobs and inference endpoints. AutoGluon 1.5 and earlier shipped separate `autogluon-training` and `autogluon-inference` images.

| Variant | Image |
| --- | --- |
| GPU | `763104351884.dkr.ecr.<region>.amazonaws.com/autogluon:1.6-cu133-amzn2023` |
| CPU | `763104351884.dkr.ecr.<region>.amazonaws.com/autogluon:1.6-cpu-amzn2023` |

The `1.6` tags always point to the latest 1.6.x patch release. To pin a patch release, use the full version (for example `1.6.3-cu133-amzn2023`). For the account IDs in other regions, see [Available Images](https://aws.github.io/deep-learning-containers/reference/available_images/#autogluon) in the DLC documentation.

## What's included

- **[AutoGluon](https://github.com/autogluon/autogluon) 1.6.3**
- **Tabular classification and regression:** [scikit-learn](https://scikit-learn.org/), [LightGBM](https://github.com/microsoft/LightGBM), [CatBoost](https://github.com/catboost/catboost), [XGBoost](https://github.com/dmlc/xgboost), [TabM](https://github.com/yandex-research/tabm)
- **Tabular foundation models:** [Mitra](https://huggingface.co/autogluon/mitra-classifier), [TabICL](https://github.com/soda-inria/tabicl), [TabDPT](https://github.com/layer6ai-labs/TabDPT), [Nori](https://github.com/synthefy/synthefy-nori)
- **Time series forecasting:** [StatsForecast](https://github.com/Nixtla/statsforecast) (statistical models such as ETS and ARIMA), [GluonTS](https://github.com/awslabs/gluonts) (deep learning models such as DeepAR, TFT, and PatchTST), [MLForecast](https://github.com/Nixtla/mlforecast)
- **Time series foundation models:** [Chronos](https://github.com/amazon-science/chronos-forecasting), [Toto 2.0](https://huggingface.co/collections/Datadog/toto-20)
- **Inference server** implementing the SageMaker `/ping` and `/invocations` contract. Your handler defines `model_fn` and `transform_fn` — see [Reference](reference.md).

The image is built on the [PyTorch DLC](https://aws.github.io/deep-learning-containers/pytorch/): PyTorch 2.13, Python 3.12, CUDA 13.3 (GPU variant), Amazon Linux 2023.

Foundation model weights are not baked into the image. They are downloaded from Hugging Face the first time a model is used, so the training job or endpoint needs internet access, or the weights must be bundled into the model artifact (see {py:meth}`~autogluon.cloud.TimeSeriesFoundationModel.cache_model_artifact`).

## How AutoGluon-Cloud uses the image

AutoGluon-Cloud picks the image for you based on the `framework_version` argument of `fit()`, `predict()` and `deploy()` (default `"1.6"`), the AWS region, and whether the instance type has a GPU. It then supplies its own [training and inference scripts](https://github.com/autogluon/autogluon-cloud/tree/master/src/autogluon/cloud/scripts/sagemaker_scripts) — the same kind of scripts you would write yourself following [Custom Scripts](custom-scripts.md).

To run AutoGluon-Cloud on a different image — for example, one you extended with extra packages — pass `custom_image_uri`:

```python
from autogluon.cloud import TabularCloudPredictor

cloud_predictor = TabularCloudPredictor()
cloud_predictor.fit(
    train_data="train.csv",
    predictor_init_args={"label": "class"},
    custom_image_uri="<account_id>.dkr.ecr.<region>.amazonaws.com/my-autogluon:latest",
)
```

Custom images should be built `FROM` the AutoGluon image, so that the scripts AutoGluon-Cloud ships keep working.
