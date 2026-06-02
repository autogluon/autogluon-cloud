"""Unit tests for FoundationModel: serialization, hyperparameter resolution, and deploy-time wiring of
model_artifact_uri / model_path."""

from pathlib import Path
from unittest import mock

import pytest

from autogluon.cloud.model import FoundationModel


@pytest.fixture(autouse=True)
def _stub_aws(monkeypatch):
    """Avoid touching AWS / config files during construction."""
    monkeypatch.setattr(
        "autogluon.cloud.model.foundation_model.resolve_cloud_output_path",
        lambda path, backend_name: path or "s3://stub/output",
    )
    monkeypatch.setattr(
        "autogluon.cloud.backend.backend_factory.BackendFactory.get_backend",
        lambda **kwargs: mock.MagicMock(role_arn="arn:aws:iam::0:role/stub"),
    )


def test_to_dict_minimal_emits_only_model_id():
    fm = FoundationModel("chronos-2", cloud_output_path="s3://b")
    assert fm.to_dict() == {"model_id": "chronos-2"}


def test_to_dict_includes_overrides_and_artifact_when_set():
    fm = FoundationModel(
        "chronos-2",
        cloud_output_path="s3://b",
        hyperparameters={"context_length": 256},
        model_artifact_uri="s3://b/cache/chronos-2/abc/model.tar.gz",
    )
    assert fm.to_dict() == {
        "model_id": "chronos-2",
        "hyperparameters": {"context_length": 256},
        "model_artifact_uri": "s3://b/cache/chronos-2/abc/model.tar.gz",
    }


def test_to_dict_excludes_runtime_context():
    fm = FoundationModel(
        "chronos-2",
        cloud_output_path="s3://my-bucket/runs/",
        role="arn:aws:iam::0:role/runtime",
    )
    d = fm.to_dict()
    assert "role" not in d
    assert "cloud_output_path" not in d
    assert "backend" not in d


def test_from_dict_round_trip():
    fm = FoundationModel(
        "chronos-2",
        cloud_output_path="s3://b",
        hyperparameters={"context_length": 256},
        model_artifact_uri="s3://b/cache/chronos-2/abc/model.tar.gz",
    )
    fm2 = FoundationModel.from_dict(fm.to_dict(), cloud_output_path="s3://other")
    assert fm2.to_dict() == fm.to_dict()


def test_from_json_round_trip():
    fm = FoundationModel("chronos-2", cloud_output_path="s3://b")
    fm2 = FoundationModel.from_json(fm.to_json(), cloud_output_path="s3://b")
    assert fm2.to_dict() == fm.to_dict()


def test_inference_hyperparameters_default_model_path_to_source_uri():
    fm = FoundationModel("chronos-2", cloud_output_path="s3://b")
    hp = fm._get_hyperparameters("inference")
    assert hp["model_path"] == "autogluon/chronos-2"


def test_user_hyperparameter_override_wins_over_default_model_path():
    fm = FoundationModel(
        "chronos-2",
        cloud_output_path="s3://b",
        hyperparameters={"model_path": "my-org/my-finetune"},
    )
    hp = fm._get_hyperparameters("inference")
    assert hp["model_path"] == "my-org/my-finetune"


def test_deploy_passes_artifact_uri_and_overrides_model_path_to_container_dir():
    fm = FoundationModel(
        "chronos-2",
        cloud_output_path="s3://b",
        model_artifact_uri="s3://b/cache/chronos-2/model.tar.gz",
    )
    fm._backend.endpoint = mock.MagicMock()  # _deploy_backend asserts this is set after the call
    fm._deploy_backend()

    call = fm._backend.deploy.call_args
    assert call.kwargs["predictor_path"] == "s3://b/cache/chronos-2/model.tar.gz"
    assert call.kwargs["repack"] is False
    serve_cfg = call.kwargs["fm_serve_config"]
    assert serve_cfg["hyperparameters"]["model_path"] == "/opt/ml/model/weights"


def test_deploy_without_artifact_passes_none_predictor_path_and_source_uri():
    fm = FoundationModel("chronos-2", cloud_output_path="s3://b")
    fm._backend.endpoint = mock.MagicMock()
    fm._deploy_backend()

    call = fm._backend.deploy.call_args
    assert call.kwargs["predictor_path"] is None
    assert call.kwargs["repack"] is False
    serve_cfg = call.kwargs["fm_serve_config"]
    assert serve_cfg["hyperparameters"]["model_path"] == "autogluon/chronos-2"


def test_deploy_rejects_user_model_path_when_artifact_uri_set():
    """User-supplied model_path is incoherent with model_artifact_uri (the bundled tarball dictates the in-container
    path). Raise rather than silently overwrite."""
    fm = FoundationModel(
        "chronos-2",
        cloud_output_path="s3://b",
        model_artifact_uri="s3://b/cache/chronos-2/model.tar.gz",
    )
    fm._backend.endpoint = mock.MagicMock()
    with pytest.raises(ValueError, match="model_artifact_uri"):
        fm._deploy_backend(hyperparameters={"model_path": "my-org/something-else"})


def test_cache_model_artifact_rejects_non_s3_path():
    fm = FoundationModel("chronos-2", cloud_output_path="s3://b")
    with pytest.raises(ValueError, match="s3://"):
        fm.cache_model_artifact("/local/path")


def test_cache_model_artifact_uploads_with_version_metadata(monkeypatch):
    """On cache miss, upload_file runs with the version metadata key — that's the cache-invalidation contract."""
    from autogluon.cloud.version import __version__

    fm = FoundationModel("chronos-2", cloud_output_path="s3://b")
    s3 = mock.MagicMock()
    fm._backend.sagemaker_session.boto_session.client.return_value = s3
    monkeypatch.setattr("autogluon.cloud.model.foundation_model._s3_head_or_none", lambda *_: None)
    monkeypatch.setattr("autogluon.cloud.model.foundation_model.tarfile", mock.MagicMock())
    monkeypatch.setattr("huggingface_hub.snapshot_download", mock.MagicMock())
    monkeypatch.setattr(FoundationModel, "_serve_script_path", "/tmp/nonexistent-stub.py")
    monkeypatch.setattr(Path, "read_bytes", lambda self: b"")
    monkeypatch.setattr(Path, "write_bytes", lambda self, data: None)

    new_fm = fm.cache_model_artifact("s3://b/cache")

    assert new_fm.model_artifact_uri == "s3://b/cache/chronos-2/model.tar.gz"
    s3.upload_file.assert_called_once()
    metadata = s3.upload_file.call_args.kwargs["ExtraArgs"]["Metadata"]
    assert metadata == {"autogluon-cloud-version": __version__}


def test_cache_model_artifact_raises_on_stale_version_without_overwrite():
    """Returning a model pointing at a tarball bundled by a different autogluon-cloud version surfaces as a confusing
    endpoint failure later. Force the user to opt in."""
    fm = FoundationModel("chronos-2", cloud_output_path="s3://b")
    s3 = mock.MagicMock()
    s3.head_object.return_value = {"Metadata": {"autogluon-cloud-version": "0.0.0-stale"}}
    fm._backend.sagemaker_session.boto_session.client.return_value = s3

    with pytest.raises(RuntimeError, match="overwrite=True"):
        fm.cache_model_artifact("s3://b/cache")


def test_sagemaker_backend_uses_nonrepack_when_repack_is_false():
    """A pre-bundled cached artifact should bypass the SDK's download/repack/re-upload path."""
    from autogluon.cloud.backend.sagemaker_backend import SagemakerBackend

    sb = "autogluon.cloud.backend.sagemaker_backend"
    with (
        mock.patch(f"{sb}.setup_sagemaker_session", return_value=mock.MagicMock(boto_region_name="us-east-1")),
        mock.patch(f"{sb}.resolve_execution_role", return_value="arn:aws:iam::000000000000:role/t"),
        mock.patch(f"{sb}.AutoGluonNonRepackInferenceModel") as nonrepack_cls,
        mock.patch(f"{sb}.AutoGluonRepackInferenceModel") as repack_cls,
        mock.patch.object(SagemakerBackend, "_upload_predictor", side_effect=lambda p, _: p),
    ):
        backend = SagemakerBackend(
            local_output_path="/tmp/t",
            cloud_output_path="s3://bucket/run",
            predictor_type="timeseries",
        )
        backend._fit_job = None
        backend.deploy(
            predictor_path="s3://bucket/cache/chronos-2/model.tar.gz",
            endpoint_name="ep",
            model_kwargs={"entry_point": "stub.py"},
            repack=False,
        )

        nonrepack_cls.assert_called_once()
        repack_cls.assert_not_called()
