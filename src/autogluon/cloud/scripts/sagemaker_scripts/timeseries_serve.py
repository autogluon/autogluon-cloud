# flake8: noqa
import json
import os
import shutil

import pandas as pd

from autogluon.timeseries import TimeSeriesPredictor

from timeseries_serve_utils import (
    APPLICATION_JSON,
    X_AUTOGLUON,
    parse_dataframe_payload,
    parse_jumpstart_payload,
    parse_x_autogluon_payload,
    render_dataframe,
    render_jumpstart,
)


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
    id_column = model._id_column
    timestamp_column = model._timestamp_column

    if input_content_type == X_AUTOGLUON:
        tsdf, known_covariates, inference_kwargs = parse_x_autogluon_payload(
            request_body, id_column=id_column, timestamp_column=timestamp_column
        )
    elif input_content_type == APPLICATION_JSON:
        tsdf, known_covariates, inference_kwargs = parse_jumpstart_payload(
            request_body, target_column=model.target, id_column=id_column, timestamp_column=timestamp_column
        )
    else:
        tsdf, known_covariates, inference_kwargs = parse_dataframe_payload(
            request_body, input_content_type, id_column=id_column, timestamp_column=timestamp_column
        )

    predictions = model.predict(tsdf, known_covariates=known_covariates, **inference_kwargs)

    if input_content_type == APPLICATION_JSON:
        return render_jumpstart(
            predictions.to_data_frame().reset_index(), id_column=id_column, timestamp_column=timestamp_column
        )
    # Preserve legacy wire format: keep the (item_id, timestamp) multiindex on the dataframe.
    return render_dataframe(pd.DataFrame(predictions), output_content_type)
