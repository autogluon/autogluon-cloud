"""Unit tests for the training sourcedir built by upload_training_code."""

import tarfile
from unittest import mock

import pytest

from autogluon.cloud.utils.ag_sagemaker import upload_training_code


def _sourcedir_contents(tmp_path, requirements):
    entry_point = tmp_path / "train.py"
    entry_point.write_text("")
    contents = {}

    def read_tarball(path, bucket, key_prefix):
        with tarfile.open(path) as tar:
            for member in tar.getmembers():
                f = tar.extractfile(member)
                contents[member.name] = f.read().decode() if f is not None else None
        return f"s3://{bucket}/{key_prefix}/sourcedir.tar.gz"

    session = mock.MagicMock()
    session.upload_data.side_effect = read_tarball
    uri = upload_training_code(str(entry_point), session, "s3://b/code/job/source", requirements=requirements)
    assert uri == "s3://b/code/job/source/sourcedir.tar.gz"
    return contents


@pytest.mark.parametrize("requirements", [None, []])
def test_sourcedir_without_requirements(tmp_path, requirements):
    contents = _sourcedir_contents(tmp_path, requirements)
    assert "train.py" in contents
    assert "serving_utils/mitra.py" in contents
    assert "requirements.txt" not in contents


def test_sourcedir_writes_requirements(tmp_path):
    contents = _sourcedir_contents(tmp_path, ["mitra-finetune==0.3.0", "foo>=1"])
    assert contents["requirements.txt"] == "mitra-finetune==0.3.0\nfoo>=1\n"
