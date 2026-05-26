# isort: skip_file
# flake8: noqa
# The import order of autogluon sub module here could cause seg fault. Ignore isort for now
# https://github.com/autogluon/autogluon/issues/2042
import argparse
import os
import pandas as pd
import shutil
from pprint import pprint

import pickle

from autogluon.common.loaders import load_pd
from autogluon.tabular import TabularPredictor, TabularDataset


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
    parser.add_argument("--train_dir", type=str, default=get_env_if_present("SM_CHANNEL_TRAIN"))
    parser.add_argument("--tune_dir", type=str, required=False, default=get_env_if_present("SM_CHANNEL_TUNE"))
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
    elif predictor_type == "timeseries":
        from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

        predictor_cls = TimeSeriesPredictor
        # Disable prediction caching to avoid errors on read-only filesystem
        predictor_init_args.setdefault("cache_predictions", False)

    id_column = ag_args.get("id_column", "item_id")
    timestamp_column = ag_args.get("timestamp_column", "timestamp")

    static_features_df = None
    if args.static_features:
        sf_file = get_input_path(args.static_features)
        static_features_df = load_pd.load(sf_file)
        if id_column in static_features_df.columns:
            static_features_df.set_index(id_column, inplace=True)

    train_file = get_input_path(args.train_dir)
    if predictor_type == "timeseries":
        training_data = TimeSeriesDataFrame.from_data_frame(
            load_pd.load(train_file),
            id_column=id_column,
            timestamp_column=timestamp_column,
            static_features_df=static_features_df,
        )
    else:
        training_data = TabularDataset(train_file)

    if predictor_type == "tabular" and "image_column" in ag_args:
        feature_metadata = predictor_fit_args.get("feature_metadata", None)
        assert feature_metadata is not None, (
            f"Detected image_column: {ag_args['image_column']} while feature metadata is not included"
        )
        feature_metadata = feature_metadata.add_special_types({ag_args["image_column"]: ["image_path"]})
        predictor_fit_args["feature_metadata"] = feature_metadata

    tuning_data = None
    if args.tune_dir:
        tune_file = get_input_path(args.tune_dir)
        if predictor_type == "timeseries":
            tuning_data = TimeSeriesDataFrame.from_data_frame(
                load_pd.load(tune_file),
                id_column=id_column,
                timestamp_column=timestamp_column,
                static_features_df=static_features_df,
            )
        else:
            tuning_data = TabularDataset(tune_file)

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

    known_covariates_df = None
    if args.known_covariates:
        kc_file = get_input_path(args.known_covariates)
        known_covariates_df = load_pd.load(kc_file)
        if "known_covariates_names" not in predictor_init_args:
            kc_cols = known_covariates_df.columns.to_list()
            predictor_init_args["known_covariates_names"] = [
                c for c in kc_cols if c not in (id_column, timestamp_column)
            ]

    predictor = predictor_cls(**predictor_init_args).fit(training_data, tuning_data=tuning_data, **predictor_fit_args)

    # When use automm backend, predictor needs to be saved with standalone flag to avoid need of internet access when loading
    # This is required because of https://discuss.huggingface.co/t/error-403-when-downloading-model-for-sagemaker-batch-inference/12571/6
    if predictor_type == "multimodal":
        predictor.save(path=save_path, standalone=True)

    if predictor_cls == TabularPredictor:
        if ag_args.get("leaderboard", False):
            lb = predictor.leaderboard(silent=False)
            lb.to_csv(f"{args.output_data_dir}/leaderboard.csv")

    if ag_args.get("predict_after_fit", False):
        if predictor_type != "timeseries":
            raise NotImplementedError(
                f"`fit_predict` is only supported for predictor_type='timeseries', got '{predictor_type}'."
            )
        print("Running in-job prediction for fit_predict")
        known_covariates_tsdf = None
        if known_covariates_df is not None:
            known_covariates_tsdf = TimeSeriesDataFrame.from_data_frame(
                known_covariates_df, id_column=id_column, timestamp_column=timestamp_column
            )
        predictions = predictor.predict(training_data, known_covariates=known_covariates_tsdf)
        predictions = pd.DataFrame(predictions).reset_index()
        predictions_path = os.path.join(args.output_data_dir, "predictions.csv")
        predictions.to_csv(predictions_path, index=False)
        print(f"Saved predictions to {predictions_path}")

    print("Saving serving script")
    serving_script_saving_path = os.path.join(save_path, "code")
    os.mkdir(serving_script_saving_path)
    serving_script_path = get_input_path(args.serving_script)
    shutil.move(serving_script_path, os.path.join(serving_script_saving_path, os.path.basename(serving_script_path)))
