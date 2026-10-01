#!/bin/bash

MODULE=$1
AG_VERSION="${2:-source}"

set -ex

source $(dirname "$0")/env_setup.sh

install_cloud_test

extra_args=()
if [[ "$MODULE" == "tabular" || "$MODULE" == "timeseries" ]]; then
    # Train once, then let the parallel follow-up tests attach to the completed job by name.
    export AG_CLOUD_SHARED_TRAINING_JOB_NAME="ag-cloud-ci-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-${MODULE}-${AG_VERSION//./-}"
    train_test="tests/unittests/$MODULE/test_$MODULE.py::test_${MODULE}_train"
    python3 -m pytest "$train_test" --framework_version $AG_VERSION
    extra_args+=(--deselect "$train_test")
fi

python3 -m pytest -n 4 --junitxml=results.xml tests/unittests/$MODULE/ --framework_version $AG_VERSION "${extra_args[@]}"
