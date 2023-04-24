import os
from pathlib import Path

from ..backend.constant import MULTIMODL_SAGEMAKER, TABULAR_RAY_AWS, TABULAR_SAGEMAKER, TIMESERIES_SAGEMAKER


class ScriptManager:
    CLOUD_PATH = Path(__file__).parent.parent.absolute()
    SCRIPTS_PATH = os.path.join(CLOUD_PATH, "scripts")
    SAGEMAKER_SCRIPTS_PATH = os.path.join(SCRIPTS_PATH, "sagemaker_scripts")
    RAY_SCRIPTS_PATH = os.path.join(SCRIPTS_PATH, "ray_scripts")
    SAGEMAKER_TRAIN_SCRIPT_PATH = os.path.join(SAGEMAKER_SCRIPTS_PATH, "train.py")
    SAGEMAKER_TABULAR_SERVE_SCRIPT_PATH = os.path.join(SAGEMAKER_SCRIPTS_PATH, "tabular_serve.py")
    SAGEMAKER_MULTIMODAL_SERVE_SCRIPT_PATH = os.path.join(SAGEMAKER_SCRIPTS_PATH, "multimodal_serve.py")
    SAGEMAKER_TIMESERIES_SERVE_SCRIPT_PATH = os.path.join(SAGEMAKER_SCRIPTS_PATH, "timeseries_serve.py")
    RAY_TABULAR_TRAIN_SCRIPT_PATH = os.path.join(RAY_SCRIPTS_PATH, "train.py")
    _BACKEND_SERVE_SCRIPT_MAP = {
        TABULAR_SAGEMAKER: SAGEMAKER_TABULAR_SERVE_SCRIPT_PATH,
        MULTIMODL_SAGEMAKER: SAGEMAKER_MULTIMODAL_SERVE_SCRIPT_PATH,
        TIMESERIES_SAGEMAKER: SAGEMAKER_TIMESERIES_SERVE_SCRIPT_PATH,
    }
    _BACKEND_TRAIN_MAP = {
        TABULAR_SAGEMAKER: SAGEMAKER_TRAIN_SCRIPT_PATH,
        MULTIMODL_SAGEMAKER: SAGEMAKER_TRAIN_SCRIPT_PATH,
        TIMESERIES_SAGEMAKER: SAGEMAKER_TRAIN_SCRIPT_PATH,
        TABULAR_RAY_AWS: RAY_TABULAR_TRAIN_SCRIPT_PATH,
    }

    @classmethod
    def get_train_script(cls, backend_type, framework_version):
        assert backend_type in cls._BACKEND_TRAIN_MAP
        # tabular, multimodal, timeseries share the same training script
        return cls._BACKEND_TRAIN_MAP[backend_type]

    @classmethod
    def get_serve_script(cls, backend_type, framework_version):
        assert backend_type in cls._BACKEND_SERVE_SCRIPT_MAP
        return cls._BACKEND_SERVE_SCRIPT_MAP[backend_type]
