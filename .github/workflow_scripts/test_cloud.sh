#!/bin/bash

MODULE=$1

set -ex

source $(dirname "$0")/env_setup.sh

setup_build_env
install_cloud

python3 -m pytest -n 2 --junitxml=results.xml tests/unittests/$MODULE/
