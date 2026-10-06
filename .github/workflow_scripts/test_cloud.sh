#!/bin/bash

MODULE=$1
# Pass "release" to run only the pre-release tests (every registry model) instead of the regular suite.
MARKER=${2:-"not release"}
AG_VERSION="latest"

set -ex

source $(dirname "$0")/env_setup.sh

install_cloud_test

extra_args=()
if [[ "$MODULE" != "general" && "$MARKER" != "release" ]]; then
    # Train once, then let the parallel follow-up tests attach to the completed job by name.
    export AG_CLOUD_SHARED_TRAINING_JOB_NAME="ag-cloud-ci-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-${MODULE}-${AG_VERSION//./-}"
    train_test="tests/unittests/$MODULE/test_$MODULE.py::test_${MODULE}_train"
    python3 -m pytest "$train_test" --framework_version $AG_VERSION
    extra_args+=(--deselect "$train_test")
fi

# Workers mostly wait on SageMaker, so run roughly one per test: wall time is then bounded by the slowest test.
python3 -m pytest -n 24 --durations=0 --junitxml=results.xml tests/unittests/$MODULE/ --framework_version $AG_VERSION -m "$MARKER" "${extra_args[@]}"
