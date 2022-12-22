#!/bin/bash

MODULE=$1
AG_VERSION="${2:-latest}"

set -ex

source $(dirname "$0")/env_setup.sh

if [ MODULE == "tabular" ]
then
    install_tabular $AG_VERSION
elif [ MODULE == "multimodal" ]
then
    install_multimodal $AG_VERSION
fi

install_cloud

python3 -m pytest -n 2 --junitxml=results.xml tests/unittests/$MODULE/ --framework_version $AG_VERSION
