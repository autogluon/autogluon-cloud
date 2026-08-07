#!/bin/bash

MODULE=$1
AG_VERSION="${2:-source}"

set -ex

source $(dirname "$0")/env_setup.sh

install_cloud_test

python3 -m pytest -n 4 --junitxml=results.xml tests/unittests/$MODULE/ --framework_version $AG_VERSION
