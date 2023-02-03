import os
from pathlib import Path


class ScriptManager:
    CLOUD_PATH = Path(__file__).parent.parent.absolute()
    SCRIPTS_PATH = os.path.join(CLOUD_PATH, "scripts")
    TRAIN_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, "train.py")
    TABULAR_SERVE_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, "tabular_serve.py")
    MULTIMODAL_SERVE_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, "multimodal_serve.py")
    TIMESERIES_SERVE_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, "timeseries_serve.py")
    _SERVE_SCRIPT_MAP = dict(
        tabular=TABULAR_SERVE_SCRIPT_PATH,
        multimodal=MULTIMODAL_SERVE_SCRIPT_PATH,
        timeseries=TIMESERIES_SERVE_SCRIPT_PATH,
    )

    @classmethod
    def get_train_script(cls, predictor_type, framework_version):
        assert predictor_type in ["tabular", "multimodal", "timeseries"]
        # tabular, multimodal, timeseries share the same training script
        return cls.TRAIN_SCRIPT_PATH

    @classmethod
    def get_serve_script(cls, predictor_type, framework_version):
        assert predictor_type in ["tabular", "multimodal", "timeseries"]
        return cls._SERVE_SCRIPT_MAP[predictor_type]
