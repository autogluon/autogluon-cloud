# flake8: noqa
import json
import os
import shutil

from autogluon.timeseries import TimeSeriesPredictor

from serving_utils.timeseries import parse_payload, render_response


def model_fn(model_dir):
    """loads model from previously saved artifact"""
    # TSPredictor will write to the model file during inference while the default model_dir is read only
    # Copy the model file to a writable location as a temporary workaround
    tmp_model_dir = os.path.join("/tmp", "model")
    try:
        shutil.copytree(model_dir, tmp_model_dir, dirs_exist_ok=False)
    except:
        # model already copied
        pass
    model = TimeSeriesPredictor.load(tmp_model_dir)
    if hasattr(model, "persist"):  # timeseries added persist in v1.1
        model.persist()

    metadata_path = os.path.join(tmp_model_dir, "predictor_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        model._id_column = metadata["id_column"]
        model._timestamp_column = metadata["timestamp_column"]
    else:
        model._id_column = "item_id"
        model._timestamp_column = "timestamp"
    return model


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    # prediction_length / quantile_levels are baked into the predictor at fit time, so
    # any "parameters" block in a JumpStart payload is parsed but not applied.
    tsdf, known_covariates, _ = parse_payload(
        request_body, input_content_type, id_column=model._id_column, timestamp_column=model._timestamp_column
    )
    predictions = model.predict(tsdf, known_covariates=known_covariates)
    return render_response(predictions, output_content_type)
