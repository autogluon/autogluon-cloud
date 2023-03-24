#!/bin/bash

MODULE=$1
AG_VERSION="${2:-source}"

set -ex

source $(dirname "$0")/env_setup.sh

install_cloud_test

if [ $MODULE = "tabular" ]
then
    install_tabular $AG_VERSION
elif [ $MODULE = "multimodal" ]
then
    install_multimodal $AG_VERSION
fi

python3 -m pytest -n 2 --junitxml=results.xml tests/unittests/$MODULE/ --framework_version $AG_VERSION
