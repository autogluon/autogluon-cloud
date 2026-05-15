"""Serve script for tabular foundation models (Mitra, etc.) deployed to SageMaker endpoints.

Tabular FMs differ from standard TabularPredictor endpoints:
- They need both train_data (few-shot context) and test_data per request
- There's no pre-fitted predictor on disk — the model downloads weights at startup
- Each request provides labeled context + unlabeled test rows
"""

import pickle
from io import BytesIO

import pandas as pd


def model_fn(model_dir):
    """Load foundation model config and initialize.

    The model tarball contains a config file with:
    - model_name: AG model class name (e.g., "Mitra")
    - model_path: HuggingFace model ID or S3 path to cached weights
    - default_hyperparameters: base hyperparameters from the registry

    Weights are downloaded from HuggingFace at this point.
    """
    # TODO: load config from model_dir
    # TODO: download model weights and initialize
    # Unlike timeseries, tabular FMs may not need a predictor.fit() at startup —
    # they do in-context learning per request.
    raise NotImplementedError("tabular_fm_serve.model_fn")


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    """Handle inference requests.

    Expected payload format (application/x-autogluon):
        pickle({
            "train_data": parquet_bytes,     # labeled few-shot context
            "test_data": parquet_bytes,       # unlabeled rows to predict
            "label": "target",               # target column name
            "inference_kwargs": {            # per-request overrides
                "predict_proba": False,
            },
        })

    For simpler formats, the split between train/test is not possible —
    application/x-autogluon is the primary supported format for tabular FMs.
    """
    if input_content_type == "application/x-autogluon":
        buf = bytes(request_body)
        payload = pickle.loads(buf)
        train_data = pd.read_parquet(BytesIO(payload["train_data"]))  # noqa: F841
        test_data = pd.read_parquet(BytesIO(payload["test_data"]))  # noqa: F841
        label = payload.get("label", "target")  # noqa: F841
        inference_kwargs = payload.get("inference_kwargs", {})  # noqa: F841
    else:
        raise ValueError(
            f"{input_content_type} not supported for tabular foundation models. "
            "Use 'application/x-autogluon' format to provide both train_data and test_data."
        )

    # TODO: run prediction
    # predictor = TabularPredictor(label=label).fit(
    #     train_data, hyperparameters={model.model_name: model.hyperparameters}
    # )
    # if inference_kwargs.get("predict_proba", False):
    #     prediction = predictor.predict_proba(test_data)
    # else:
    #     prediction = predictor.predict(test_data)

    # TODO: serialize output
    # if "application/x-parquet" in output_content_type:
    #     output = prediction.to_parquet()
    # elif "text/csv" in output_content_type:
    #     output = prediction.to_csv(index=None)
    # ...

    raise NotImplementedError("tabular_fm_serve.transform_fn")
