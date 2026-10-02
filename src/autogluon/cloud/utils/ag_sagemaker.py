"""Packaging helpers for AutoGluon training and serving code on SageMaker.

These replace the SageMaker SDK v2 ``Estimator`` / ``Model`` script-mode machinery: the AutoGluon DLCs still run
the SageMaker training and inference toolkits, which locate user code through the ``sagemaker_program`` /
``sagemaker_submit_directory`` hyperparameters (training) and the ``SAGEMAKER_PROGRAM`` /
``SAGEMAKER_SUBMIT_DIRECTORY`` environment variables (inference).
"""

import json
import os
import shutil
import tarfile
import tempfile
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix

from .dlc_utils import retrieve_image_uri
from .utils import safe_unpack_archive

SOURCE_DIR_TARBALL_NAME = "sourcedir.tar.gz"


def resolve_image_uri(
    image_uri: Optional[str],
    framework_version: Optional[str],
    py_version: Optional[str],
    region: str,
    image_scope: str,
    instance_type: str,
) -> str:
    """Return ``image_uri`` if set, otherwise the official AutoGluon DLC for the given version and instance."""
    if image_uri:
        return image_uri
    return retrieve_image_uri(
        framework_version=framework_version,
        region=region,
        image_scope=image_scope,
        instance_type=instance_type,
        py_version=py_version,
    )


def upload_training_code(entry_point: str, sagemaker_session, s3_uri_prefix: str) -> str:
    """Bundle the training entry point as ``sourcedir.tar.gz`` and upload it.

    Returns the S3 URI of the uploaded tarball.
    """
    with tempfile.TemporaryDirectory(prefix="ag_train_code_") as tmpdir:
        tarball_path = os.path.join(tmpdir, SOURCE_DIR_TARBALL_NAME)
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(entry_point, arcname=os.path.basename(entry_point))
        bucket, key_prefix = s3_path_to_bucket_prefix(s3_uri_prefix)
        return sagemaker_session.upload_data(path=tarball_path, bucket=bucket, key_prefix=key_prefix)


def training_script_hyperparameters(
    entry_point: str, submit_directory: str, job_name: str, region: str
) -> Dict[str, str]:
    """Hyperparameters the SageMaker training toolkit uses to download and run the entry point.

    Values are JSON-encoded, matching what SageMaker SDK v2 sent; the toolkit JSON-decodes them.
    """
    return {
        "sagemaker_program": json.dumps(os.path.basename(entry_point)),
        "sagemaker_submit_directory": json.dumps(submit_directory),
        "sagemaker_container_log_level": json.dumps(20),
        "sagemaker_job_name": json.dumps(job_name),
        "sagemaker_region": json.dumps(region),
    }


def script_mode_environment(entry_point: str, region: str) -> Dict[str, str]:
    """Environment variables pointing the inference toolkit at the serve script under the model's ``code/`` dir."""
    return {
        "SAGEMAKER_PROGRAM": os.path.basename(entry_point),
        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
        "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
        "SAGEMAKER_REGION": region,
    }


@contextmanager
def staged_serving_code(entry_point: str) -> Iterator[str]:
    """Yield a temporary directory holding ``entry_point`` and ``serving_utils/``, i.e. the model's ``code/`` dir."""
    from ..scripts import ScriptManager  # deferred: importing scripts pulls in the backend package

    staging_dir = tempfile.mkdtemp(prefix="ag_serving_")
    try:
        shutil.copy(entry_point, os.path.join(staging_dir, os.path.basename(entry_point)))
        shutil.copytree(ScriptManager.SAGEMAKER_SERVING_UTILS_DIR, os.path.join(staging_dir, "serving_utils"))
        yield staging_dir
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def repack_model_with_serving_code(
    model_data: str,
    entry_point: str,
    repacked_model_uri: str,
    sagemaker_session,
    kms_key: Optional[str] = None,
) -> str:
    """Replace ``code/`` inside the S3 ``model_data`` tarball with ``entry_point`` + ``serving_utils/`` and upload it.

    Returns ``repacked_model_uri``.
    """
    s3 = sagemaker_session.s3_client
    with tempfile.TemporaryDirectory(prefix="ag_repack_") as tmpdir:
        original_tarball = os.path.join(tmpdir, "original.tar.gz")
        s3.download_file(*s3_path_to_bucket_prefix(model_data), original_tarball)
        model_dir = os.path.join(tmpdir, "model")
        safe_unpack_archive(original_tarball, model_dir)
        code_dir = os.path.join(model_dir, "code")
        shutil.rmtree(code_dir, ignore_errors=True)
        with staged_serving_code(entry_point) as staging_dir:
            shutil.copytree(staging_dir, code_dir)

        repacked_tarball = os.path.join(tmpdir, "model.tar.gz")
        with tarfile.open(repacked_tarball, "w:gz") as tar:
            for name in sorted(os.listdir(model_dir)):
                tar.add(os.path.join(model_dir, name), arcname=name)
        extra_args = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": kms_key} if kms_key else None
        s3.upload_file(repacked_tarball, *s3_path_to_bucket_prefix(repacked_model_uri), ExtraArgs=extra_args)
    return repacked_model_uri
