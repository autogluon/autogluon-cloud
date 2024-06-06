import os
import tempfile

from moto import mock_aws

from autogluon.cloud.utils.ec2 import _get_key_pair, create_key_pair, delete_key_pair, get_latest_ami


@mock_aws
def test_key_pair():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        create_key_pair(key_name="dummy", local_path=temp_dir)
        assert _get_key_pair(key_name="dummy") is not None
        delete_key_pair(key_name="dummy", local_path=temp_dir)
        assert _get_key_pair(key_name="dummy") is None
        assert not os.path.exists(os.path.join(temp_dir, "dummy.pem"))


def test_get_latest_ami():
    ami_id = get_latest_ami()
    assert ami_id is not None
