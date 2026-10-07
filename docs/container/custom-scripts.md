# Train and Serve Custom Scripts

Use the [AutoGluon container](index.md) to run your own scripts in SageMaker training jobs, and to serve the resulting models from SageMaker endpoints. This is the same mechanism AutoGluon-Cloud uses under the hood — use it directly when you need full control over the training or inference code.

```{tip}
If you just want to train a predictor or run a foundation model, AutoGluon-Cloud already ships the scripts for you — see [Train Your Own Predictor](../tutorials/predictor.md) and [Foundation Models](../tutorials/foundation-model.md).
```

## How it works

- **Training.** A SageMaker training job runs any script you provide, using AutoGluon or any of the bundled packages. Data channels, hyperparameters and the output directory reach your script through the standard [SageMaker training toolkit](https://github.com/aws/sagemaker-training-toolkit) environment (`SM_CHANNEL_*`, `SM_MODEL_DIR`, `SM_NUM_GPUS`, ...). See [How SageMaker runs your training image](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html) for details.
- **Inference.** The endpoint loads a `model.tar.gz` that contains your model files and an inference handler under `code/`:

  ```text
  model.tar.gz
  ├── <your model files>
  └── code/
      ├── inference.py        # inference handler
      └── requirements.txt    # optional, installed at container start
  ```

  The simplest way to produce it is to have your training script save the model and copy `code/` into `SM_MODEL_DIR`. To use a different handler file name, set the `SAGEMAKER_PROGRAM` environment variable on the model.

The handler defines two functions:

```python
from typing import Any


def model_fn(model_dir: str) -> Any:
    """Load the model from /opt/ml/model. Called once at startup."""


def transform_fn(
    model: Any,
    request_body: str | bytes,
    input_content_type: str,
    output_content_type: str,
) -> tuple[str | bytes, str]:
    """Make predictions for one request. Return (response_body, response_content_type)."""
```

See [Reference](reference.md) for the full handler contract and the environment variables the server reads.

## Example: AutoGluon-Tabular

This example trains a `TabularPredictor`, deploys it to a real-time endpoint and sends it a request. The same pattern works for `TimeSeriesPredictor` or any other library in the image. It uses the SageMaker Python SDK v3 (`sagemaker>=3.0,<4`).

The project has two files:

```text
code/
├── train.py        # training entry script
└── inference.py    # inference handler, copied into the model artifact by train.py
```

### Train

`train.py` fits the predictor and copies the `code/` directory, including `inference.py`, into the model artifact:

```python
# code/train.py
import argparse
import os
import shutil

from autogluon.tabular import TabularPredictor

parser = argparse.ArgumentParser()
parser.add_argument("--time_limit", type=int, default=600)
args = parser.parse_args()

model_dir = os.environ["SM_MODEL_DIR"]
predictor = TabularPredictor(label="class", path=model_dir)
predictor.fit(
    train_data=os.environ["SM_CHANNEL_TRAIN"] + "/train.csv",
    time_limit=args.time_limit,
)

# Package this directory (including inference.py) with the model
shutil.copytree(os.path.dirname(os.path.abspath(__file__)), model_dir + "/code", dirs_exist_ok=True)
```

`inference.py` loads the predictor and returns predictions as JSON. Write it before launching the training job, since it is packaged at training time:

```python
# code/inference.py
from io import StringIO

import pandas as pd
from autogluon.tabular import TabularPredictor


def model_fn(model_dir):
    return TabularPredictor.load(model_dir)


def transform_fn(model, request_body, input_content_type, output_content_type):
    data = pd.read_json(StringIO(request_body))
    predictions = model.predict(data)
    return predictions.to_json(orient="records"), "application/json"
```

Launch the training job:

```python
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.training.configs import Compute, InputData, SourceCode
from sagemaker.train import ModelTrainer

session = Session()
ROLE_ARN = "arn:aws:iam::<account_id>:role/<SageMakerRole>"
IMAGE_URI = f"763104351884.dkr.ecr.{session.boto_region_name}.amazonaws.com/autogluon:1.6-cpu-amzn2023"

trainer = ModelTrainer(
    training_image=IMAGE_URI,
    source_code=SourceCode(source_dir="code", entry_script="train.py"),
    compute=Compute(instance_type="ml.m5.2xlarge", instance_count=1),
    hyperparameters={"time_limit": 600},
    role=ROLE_ARN,
)
trainer.train(
    input_data_config=[
        InputData(channel_name="train", data_source=session.upload_data("train.csv")),
    ],
)
print(trainer._latest_training_job.model_artifacts.s3_model_artifacts)  # s3://.../model.tar.gz
```

```{tip}
If you ran `autogluon-cloud bootstrap` (see [Setup](../tutorials/setup.md)), you can reuse the IAM role it created as `ROLE_ARN`.
```

### Deploy

Deploy the model artifact from the training job, then invoke the endpoint:

```python
import pandas as pd
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant

MODEL_DATA_URL = "s3://<bucket>/<training-job>/output/model.tar.gz"
NAME = "autogluon-tabular"

model = Model.create(
    model_name=NAME,
    primary_container=ContainerDefinition(
        image=IMAGE_URI,
        model_data_url=MODEL_DATA_URL,
    ),
    execution_role_arn=ROLE_ARN,
)
config = EndpointConfig.create(
    endpoint_config_name=NAME,
    production_variants=[
        ProductionVariant(
            variant_name="AllTraffic",
            model_name=NAME,
            instance_type="ml.m5.xlarge",
            initial_instance_count=1,
        ),
    ],
)
endpoint = Endpoint.create(endpoint_name=NAME, endpoint_config_name=NAME)
endpoint.wait_for_status("InService")

rows = pd.read_csv("train.csv").drop(columns="class").head(3)
response = endpoint.invoke(body=rows.to_json(orient="records"), content_type="application/json")
print(response.body.read().decode())
```

The endpoint is billed until you delete it:

```python
for resource in (endpoint, config, model):
    resource.delete()
```

## Tips

- **GPU.** Use the `1.6-cu133-amzn2023` image with GPU instances. GPU endpoints also need `inference_ami_version="al2023-ami-sagemaker-inference-gpu-4-1"` in the `ProductionVariant`.
- **Local testing.** Pass `training_mode=Mode.LOCAL_CONTAINER` (from `sagemaker.train.model_trainer`) and `instance_type="local_cpu"` to run the training job in Docker on your machine. Requires Docker Compose.
- **More handler examples.** AutoGluon-Cloud's own [inference handlers](https://github.com/autogluon/autogluon-cloud/tree/v0.6.0/src/autogluon/cloud/scripts/sagemaker_scripts) (`tabular_serve.py`, `timeseries_serve.py`, ...) follow the same `model_fn` / `transform_fn` contract and handle CSV, JSON and parquet payloads.
- **More training examples.** Any of the AutoGluon tutorials for [tabular](https://auto.gluon.ai/stable/tutorials/tabular/index.html) and [time series](https://auto.gluon.ai/stable/tutorials/timeseries/index.html) data can go into `train.py` with a matching `inference.py`.
