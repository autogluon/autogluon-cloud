import os
import tempfile

from moto import mock_ec2

from autogluon.cloud.utils.ec2 import _get_key_pair, create_key_pair, delete_key_pair


@mock_ec2
def test_key_pair():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        create_key_pair(key_name="dummy", local_path=temp_dir)
        assert _get_key_pair(key_name="dummy") is not None
        delete_key_pair(key_name="dummy", local_path=temp_dir)
        assert _get_key_pair(key_name="dummy") is None
        assert not os.path.exists(os.path.join(temp_dir, "dummy.pem"))
