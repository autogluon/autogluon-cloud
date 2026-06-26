# isort: skip_file
# flake8: noqa
# The import order of autogluon sub module here could cause seg fault. Ignore isort for now
# https://github.com/autogluon/autogluon/issues/2042
import argparse
import json
import os
import shutil
from pprint import pprint

import boto3
import pandas as pd
import pickle

from autogluon.common.loaders import load_pd
from autogluon.common.savers import save_pd
from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.timeseries import TimeSeriesDataFrame


def get_input_path(path):
    file = os.listdir(path)[0]
    if len(os.listdir(path)) > 1:
        raise ValueError(f"WARN: more than one file is found in {channel} directory")
    print(f"Using {file}")
    filename = f"{path}/{file}"
    return filename


def get_env_if_present(name):
    result = None
    if name in os.environ:
        result = os.environ[name]
    return result


def prepare_data(data_file, predictor_type, ag_args, static_features_df=None):
    if predictor_type == "timeseries":
        return TimeSeriesDataFrame.from_data_frame(
            load_pd.load(data_file),
            id_column=ag_args["id_column"],
            timestamp_column=ag_args["timestamp_column"],
            static_features_df=static_features_df,
        )
    else:
        return TabularDataset(data_file)


if __name__ == "__main__":
    # Disable Autotune
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    # ------------------------------------------------------------ Args parsing
    print("Starting AG")
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=get_env_if_present("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model-dir", type=str, default=get_env_if_present("SM_MODEL_DIR"))
    parser.add_argument("--n_gpus", type=str, default=get_env_if_present("SM_NUM_GPUS"))
    parser.add_argument("--train_dir", type=str, default=get_env_if_present("SM_CHANNEL_TRAIN_DATA"))
    parser.add_argument("--test_dir", type=str, required=False, default=get_env_if_present("SM_CHANNEL_TEST_DATA"))
    parser.add_argument("--tune_dir", type=str, required=False, default=get_env_if_present("SM_CHANNEL_TUNING_DATA"))
    parser.add_argument(
        "--train_images", type=str, required=False, default=get_env_if_present("SM_CHANNEL_TRAIN_IMAGES")
    )
    parser.add_argument(
        "--tune_images", type=str, required=False, default=get_env_if_present("SM_CHANNEL_TUNE_IMAGES")
    )
    parser.add_argument("--ag_args", type=str, default=get_env_if_present("SM_CHANNEL_AG_ARGS"))
    parser.add_argument("--serving_script", type=str, default=get_env_if_present("SM_CHANNEL_SERVING"))
    parser.add_argument(
        "--known_covariates", type=str, required=False, default=get_env_if_present("SM_CHANNEL_KNOWN_COVARIATES")
    )
    parser.add_argument(
        "--static_features", type=str, required=False, default=get_env_if_present("SM_CHANNEL_STATIC_FEATURES")
    )

    args, _ = parser.parse_known_args()

    print(f"Args: {args}")

    # See SageMaker-specific environment variables: https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script
    os.makedirs(args.output_data_dir, mode=0o777, exist_ok=True)

    ag_args_file = get_input_path(args.ag_args)
    with open(ag_args_file, "rb") as f:
        ag_args = pickle.load(f)  # AutoGluon-specific args

    if args.n_gpus:
        ag_args["num_gpus"] = int(args.n_gpus)

    print("Running training job with the config:")
    pprint(ag_args)

    # ---------------------------------------------------------------- Training
    save_path = os.path.normpath(args.model_dir)
    predictor_type = ag_args["predictor_type"]
    predictor_init_args = ag_args["predictor_init_args"]
    predictor_init_args["path"] = save_path
    predictor_fit_args = ag_args["predictor_fit_args"]
    valid_predictor_types = ["tabular", "multimodal", "timeseries"]
    assert predictor_type in valid_predictor_types, (
        f"predictor_type {predictor_type} not supported. Valid options are {valid_predictor_types}"
    )
    if predictor_type == "tabular":
        predictor_cls = TabularPredictor
    elif predictor_type == "multimodal":
        from autogluon.multimodal import MultiModalPredictor

        predictor_cls = MultiModalPredictor
    else:
        from autogluon.timeseries import TimeSeriesPredictor

        predictor_cls = TimeSeriesPredictor
        # Disable prediction caching to avoid errors on read-only filesystem
        predictor_init_args.setdefault("cache_predictions", False)
        ag_args.setdefault("id_column", "item_id")
        ag_args.setdefault("timestamp_column", "timestamp")

    static_features_df = None
    if args.static_features:
        static_features_df = load_pd.load(get_input_path(args.static_features))

    training_data = prepare_data(get_input_path(args.train_dir), predictor_type, ag_args, static_features_df)

    if predictor_type == "tabular" and "image_column" in ag_args:
        feature_metadata = predictor_fit_args.get("feature_metadata", None)
        assert feature_metadata is not None, (
            f"Detected image_column: {ag_args['image_column']} while feature metadata is not included"
        )
        feature_metadata = feature_metadata.add_special_types({ag_args["image_column"]: ["image_path"]})
        predictor_fit_args["feature_metadata"] = feature_metadata

    tuning_data = None
    if args.tune_dir:
        tuning_data = prepare_data(get_input_path(args.tune_dir), predictor_type, ag_args, static_features_df)

    if args.train_images:
        train_image_compressed_file = get_input_path(args.train_images)
        train_images_dir = "train_images"
        shutil.unpack_archive(train_image_compressed_file, train_images_dir)
        image_column = ag_args["image_column"]
        training_data[image_column] = training_data[image_column].apply(
            lambda path: os.path.join(train_images_dir, path)
        )

    if args.tune_images:
        tune_image_compressed_file = get_input_path(args.tune_images)
        tune_images_dir = "tune_images"
        shutil.unpack_archive(tune_image_compressed_file, tune_images_dir)
        image_column = ag_args["image_column"]
        tuning_data[image_column] = tuning_data[image_column].apply(lambda path: os.path.join(tune_images_dir, path))

    known_covariates = None
    if args.known_covariates:
        known_covariates = prepare_data(get_input_path(args.known_covariates), predictor_type, ag_args)
        if "known_covariates_names" not in predictor_init_args:
            predictor_init_args["known_covariates_names"] = list(known_covariates.columns)

    predictor = predictor_cls(**predictor_init_args).fit(training_data, tuning_data=tuning_data, **predictor_fit_args)

    # When use automm backend, predictor needs to be saved with standalone flag to avoid need of internet access when loading
    # This is required because of https://discuss.huggingface.co/t/error-403-when-downloading-model-for-sagemaker-batch-inference/12571/6
    if predictor_type == "multimodal":
        predictor.save(path=save_path, standalone=True)

    if predictor_type == "timeseries":
        # Persisted so the serve script can rebuild a TimeSeriesDataFrame from the test data
        # passed to predict / predict_real_time without the user having to re-specify the column names.
        with open(os.path.join(save_path, "predictor_metadata.json"), "w") as f:
            json.dump({"id_column": ag_args["id_column"], "timestamp_column": ag_args["timestamp_column"]}, f)

    if predictor_cls == TabularPredictor:
        if ag_args.get("leaderboard", False):
            lb = predictor.leaderboard(silent=False)
            lb.to_csv(f"{args.output_data_dir}/leaderboard.csv")

    if ag_args.get("predict_after_fit", False):
        print("Running in-job prediction for fit_predict")
        if predictor_type == "timeseries":
            predictions = predictor.predict(training_data, known_covariates=known_covariates)
            predictions = predictions.to_data_frame().reset_index()
        elif predictor_type == "tabular":
            if "image_column" in ag_args:
                raise NotImplementedError(
                    "`fit_predict` does not support image columns yet. "
                    "Use `fit` + `predict` for tabular data with image columns."
                )
            assert args.test_dir is not None, "`test_data` channel is required for tabular fit_predict."
            test_data = prepare_data(get_input_path(args.test_dir), predictor_type, ag_args)
            # Duplicated from tabular_serve.py:88-93 (tracked tech debt to de-dup later).
            from autogluon.core.constants import QUANTILE, REGRESSION
            from autogluon.core.utils import get_pred_from_proba_df

            if predictor.problem_type not in [REGRESSION, QUANTILE]:
                pred_proba = predictor.predict_proba(test_data, as_pandas=True)
                pred = get_pred_from_proba_df(pred_proba, problem_type=predictor.problem_type)
                pred_proba.columns = [str(c) + "_proba" for c in pred_proba.columns]
                pred.name = predictor.label
                predictions = pd.concat([pred, pred_proba], axis=1)
            else:
                predictions = predictor.predict(test_data, as_pandas=True).to_frame()
        else:
            raise NotImplementedError(f"`fit_predict` is not supported for predictor_type='{predictor_type}'.")
        predictions_path = ag_args["predictions_path"]
        # Save locally then upload via boto3: s3fs/fsspec are not available in the training container.
        local_path = os.path.join(args.output_data_dir, os.path.basename(predictions_path))
        save_pd.save(path=local_path, df=predictions)
        bucket, key = s3_path_to_bucket_prefix(predictions_path)
        boto3.client("s3").upload_file(local_path, bucket, key)
        print(f"Uploaded predictions to {predictions_path}")

    print("Saving serving artifacts")
    shutil.copytree(args.serving_script, os.path.join(save_path, "code"))
