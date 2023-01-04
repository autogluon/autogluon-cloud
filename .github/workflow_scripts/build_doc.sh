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
    bucket='autogluon-cloud-doc-staging'
    if [[ -n $PR_NUMBER ]]; then path=$PR_NUMBER; else path=$BRANCH; fi
    site=$bucket.s3-website-us-west-2.amazonaws.com/$path/$COMMIT_SHA  # site is the actual bucket location that will serve the doc
else
    if [[ $BRANCH == "master" ]]
    then
        path = "dev"
    elif [[ $BRANCH == "stable" ]]
    then
        path = "stable"
    else
        exit 0  # For other branch pushed to autogluon-cloud. We do not build docs.
    fi
    bucket='autogluon.mxnet.io'
    site=$bucket/$path  # site is the actual bucket location that will serve the doc
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

cd docs && d2lbook build html

COMMAND_EXIT_CODE=$?
if [[ $COMMAND_EXIT_CODE -ne 0 ]]; then
    exit COMMAND_EXIT_CODE
fi

if [[ (-n $PR_NUMBER) || ($GIT_REPO != "autogluon/autogluon-cloud") ]]
then
    # If PR, move the whole doc folder (to keep css styles) to staging bucket for visibility
    DOC_PATH=_build/html/
    S3_PATH=s3://$BUCKET/$path/$COMMIT_SHA
    write_to_s3 $BUCKET $DOC_PATH $S3_PATH
else
    # If master/stable, move the individual tutorial html to dev/stable bucket of main AG
    cacheControl='--cache-control max-age=7200'
    DOC_PATH=_build/html/tutorials/autogluon-cloud.html
    S3_PATH=s3://$BUCKET/$path/tutorials/cloud_fit_deploy/
    aws s3 cp $DOC_PATH $S3_PATH --acl public-read ${cacheControl}
fi
