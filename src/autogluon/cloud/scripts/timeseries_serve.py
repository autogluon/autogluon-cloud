# flake8: noqa
import os
import shutil
from io import BytesIO, StringIO

import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


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
    print("MODEL LOADED")
    return model


def prepare_timeseries_dataframe(df, predictor):
    target = predictor.target
    cols = df.columns.to_list()
    id_column = cols[0]
    timestamp_column = cols[1]
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    static_features = None
    if target != cols[-1]:
        # target is not the last column, then there are static features being merged in
        target_index = cols.index(target)
        static_columns = cols[target_index + 1 :]
        static_features = df[[id_column] + static_columns].groupby([id_column], sort=False).head(1)
        static_features.set_index(id_column, inplace=True)
        df.drop(columns=static_columns, inplace=True)
    df = TimeSeriesDataFrame.from_data_frame(df, id_column=id_column, timestamp_column=timestamp_column)
    if static_features is not None:
        df.static_features = static_features
    return df


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    if input_content_type == "application/x-parquet":
        buf = BytesIO(request_body)
        data = pd.read_parquet(buf)

    elif input_content_type == "text/csv":
        buf = StringIO(request_body)
        data = pd.read_csv(buf)

    elif input_content_type == "application/json":
        buf = StringIO(request_body)
        data = pd.read_json(buf)

    elif input_content_type == "application/jsonl":
        buf = StringIO(request_body)
        data = pd.read_json(buf, orient="records", lines=True)

    else:
        raise ValueError(f"{input_content_type} input content type not supported.")

    data = prepare_timeseries_dataframe(data, model)
    prediction = model.predict(data)
    prediction = pd.DataFrame(prediction)

    if "application/x-parquet" in output_content_type:
        prediction.columns = prediction.columns.astype(str)
        output = prediction.to_parquet()
        output_content_type = "application/x-parquet"
    elif "application/json" in output_content_type:
        output = prediction.to_json()
        output_content_type = "application/json"
    elif "text/csv" in output_content_type:
        output = prediction.to_csv(index=None)
        output_content_type = "text/csv"
    else:
        raise ValueError(f"{output_content_type} content type not supported")

    return output, output_content_type
