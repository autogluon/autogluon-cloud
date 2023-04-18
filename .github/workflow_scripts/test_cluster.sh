#!/bin/bash

AG_VERSION="${1:-source}"

set -ex

source $(dirname "$0")/env_setup.sh

install_cloud_test

python3 -m pytest --forked --junitxml=results.xml tests/unittests/cluster/ --framework_version $AG_VERSION
