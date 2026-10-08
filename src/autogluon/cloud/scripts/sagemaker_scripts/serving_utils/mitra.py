"""Patch AutoGluon's Mitra for the v2 regressor checkpoint, using the optional ``mitra-finetune`` package."""

import json
import os


def patch_mitra_regressor(hyperparameters):
    """Patch AutoGluon's Mitra to run the v2 regressor, if ``mitra_finetune`` is installed and this is a Mitra regressor."""
    from huggingface_hub import snapshot_download

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
