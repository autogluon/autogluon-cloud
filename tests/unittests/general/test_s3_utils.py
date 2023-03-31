import os
import tempfile

import boto3
from moto import mock_s3

from autogluon.cloud.utils.s3_utils import upload_file


@mock_s3
def test_upload_file():
    s3 = boto3.client("s3")
    # We need to create the bucket since this is all in Moto's 'virtual' AWS account
    bucket_name = "TestBucket"
    s3.create_bucket(Bucket=bucket_name)
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        file_name = "temp.txt"
        with open(file_name, "w") as temp_file:
            # create an empty file
            pass
        upload_file(file_name=file_name, bucket=bucket_name)
        upload_file(file_name=file_name, bucket=bucket_name, prefix="foo")
        objects = [obj["Key"] for obj in s3.list_objects(Bucket=bucket_name)["Contents"]]
        assert file_name in objects
        assert "foo/" + file_name in objects
