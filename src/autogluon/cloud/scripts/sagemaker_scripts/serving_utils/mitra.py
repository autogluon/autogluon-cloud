"""Optional-dependency install and Mitra regressor patching, shared by the tabular FM serve script."""

import json
import os
import subprocess
import sys

from huggingface_hub import snapshot_download

# TODO: drop the index override once mitra-finetune is released on PyPI.
_OPTIONAL_DEPENDENCIES_INDEX_URL = "https://test.pypi.org/simple/"


def install_optional_dependencies(specs):
    """pip-install the registry's ``optional_dependencies`` (without their dependencies) into the container."""
    if specs:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--index-url",
                _OPTIONAL_DEPENDENCIES_INDEX_URL,
                *specs,
            ]
        )


def patch_mitra_regressor(hyperparameters):
    """Patch AutoGluon's Mitra to run the v2 regressor, if ``mitra_finetune`` is installed and this is a Mitra regressor."""
    source = hyperparameters.get("hf_reg_model")
    if source is None:
        return
    try:
        from mitra_finetune.patches import install_d2h_sync_patch, install_reg_ce_patches, install_use_hf_patch
    except ImportError:
        return
    config_dir = source if os.path.isdir(source) else snapshot_download(repo_id=source, allow_patterns=["config.json"])
    with open(os.path.join(config_dir, "config.json")) as f:
        n_bins = int(json.load(f)["dim_output"])
    install_d2h_sync_patch()
    install_reg_ce_patches(n_bins)
    install_use_hf_patch()
