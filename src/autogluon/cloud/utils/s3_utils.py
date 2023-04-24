import os
from typing import Optional

import boto3
import sagemaker

from autogluon.common.utils.s3_utils import is_s3_url, s3_path_to_bucket_prefix


def upload_file(file_name: str, bucket: str, prefix: Optional[str] = None):
    """
    Upload a file to an S3 bucket

    Parameters
    ----------
    file_name: str,
        File to upload
    bucket: str,
        Bucket to upload to
    prefix: Optional[str], default = None
        S3 prefix. If not specified then will upload to the root of the bucket
    """
    object_name = os.path.basename(file_name)
    if prefix is not None and len(prefix) == 0:
        prefix = None
    if prefix is not None:
        object_name = prefix + "/" + object_name

    # Upload the file
    s3_client = boto3.client("s3")
    s3_client.upload_file(file_name, bucket, object_name)


def download_s3_file(bucket, prefix, path):
    s3 = boto3.client("s3")
    s3.download_file(bucket, prefix, path)


def is_s3_folder(path, session=None):
    """
    This function tries to determine if a s3 path is a folder.
    """
    assert is_s3_url(path)
    if session is None:
        session = sagemaker.session.Session()
    bucket, prefix = s3_path_to_bucket_prefix(path)
    contents = session.list_s3_files(bucket, prefix)
    if len(contents) > 1:
        return False
    # When the folder contains only 1 object, or the prefix is a file results in a len(contents) == 1
    # When the prefix is a file, the contents will match the prefix
    if contents[0] == prefix:
        return False
    return True
