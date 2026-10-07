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


def _check_fit_time_args(model, inference_kwargs):
    """Reject requests asking for a different prediction_length / quantile_levels / target than the predictor was
    fit with. These are fixed at fit time, so silently ignoring them would return a different forecast than asked for.
    """
    fit_time_values = {
        "prediction_length": model.prediction_length,
        "quantile_levels": sorted(model.quantile_levels),
        "target": model.target,
    }
    for key, fit_time_value in fit_time_values.items():
        if key not in inference_kwargs:
            continue
        value = inference_kwargs[key]
        if key == "quantile_levels":
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"`quantile_levels` must be a list of floats, got {value!r}.")
            value = sorted(value)
        if value != fit_time_value:
            raise ValueError(
                f"This endpoint serves a predictor fit with {key}={fit_time_value!r}, but the request has "
                f"{key}={value!r}. Omit `{key}` from the request, or fit a new predictor."
            )


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    tsdf, known_covariates, inference_kwargs = parse_payload(
        request_body,
        input_content_type,
        id_column=model._id_column,
        timestamp_column=model._timestamp_column,
        target_column=model.target,
    )
    _check_fit_time_args(model, inference_kwargs)
    predictions = model.predict(tsdf, known_covariates=known_covariates)
    return render_response(predictions, output_content_type)
