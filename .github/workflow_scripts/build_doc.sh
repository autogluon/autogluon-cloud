#!/bin/bash

BRANCH=$(basename $1)
GIT_REPO=$2
COMMIT_SHA=$3
PR_NUMBER=$4

set -ex

source $(dirname "$0")/env_setup.sh

if [[ (-n $PR_NUMBER) || ($GIT_REPO != "autogluon/autogluon-cloud") ]]
then
    # Put in cloud bucket for staging purpose
    BUCKET='autogluon-cloud-doc-staging'
    if [[ -n $PR_NUMBER ]]; then path=$PR_NUMBER; else path=$BRANCH; fi
    site=https://d12sc05jpx1wj5.cloudfront.net/$path/$COMMIT_SHA
else
    if [[ $BRANCH == "master" ]]
    then
        path="cloud/dev"
    else
        if [[ $BRANCH == "dev" ]]
        then
            path="cloud/dev-branch"
        else
            path="cloud/$BRANCH"
        fi
    fi
    BUCKET='autogluon.mxnet.io'
    site=$BUCKET/$path  # site is the actual bucket location that will serve the doc
fi

other_doc_version_text='Stable Version Documentation'
other_doc_version_branch='stable'
if [[ $BRANCH == 'stable' ]]
then
    other_doc_version_text='Dev Version Documentation'
    other_doc_version_branch='dev'
fi

setup_build_contrib_env

sed -i -e "s@###_PLACEHOLDER_WEB_CONTENT_ROOT_###@http://$site@g" docs/config.ini
sed -i -e "s@###_OTHER_VERSIONS_DOCUMENTATION_LABEL_###@$other_doc_version_text@g" docs/config.ini
sed -i -e "s@###_OTHER_VERSIONS_DOCUMENTATION_BRANCH_###@$other_doc_version_branch@g" docs/config.ini

install_cloud
cd docs && d2lbook build rst && d2lbook build html

COMMAND_EXIT_CODE=$?
if [[ $COMMAND_EXIT_CODE -ne 0 ]]; then
    exit COMMAND_EXIT_CODE
fi

if [[ (-n $PR_NUMBER) || ($GIT_REPO != "autogluon/autogluon-cloud") ]]
then
    # If PR, move the whole doc folder (to keep css styles) to staging bucket for visibility
    DOC_PATH=_build/html/
    S3_PATH=s3://$BUCKET/$path/$COMMIT_SHA
    aws s3 cp $DOC_PATH $S3_PATH --recursive
else
    # If master/stable, move the individual tutorial html to dev/stable bucket of main AG
    cacheControl='--cache-control max-age=7200'
    DOC_PATH=_build/html/tutorials/autogluon-cloud.html
    S3_PATH=s3://$BUCKET/$path/tutorials/cloud_fit_deploy/
    aws s3 cp $DOC_PATH $S3_PATH --acl public-read ${cacheControl}
fi
