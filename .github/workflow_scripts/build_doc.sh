#!/bin/bash

set -ex

source $(dirname "$0")/env_setup.sh

setup_build_contrib_env

cd docs && d2lbook build html
