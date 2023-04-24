from .backend import Backend
from .multimodal_sagemaker_backend import MultiModalSagemakerBackend
from .ray_aws_backend import TabularRayAWSBackend
from .sagemaker_backend import SagemakerBackend
from .tabular_sagemaker_backend import TabularSagemakerBackend
from .timeseries_sagemaker_backend import TimeSeriesSagemakerBackend


class BackendFactory:
    __supported_backend = [
        SagemakerBackend,
        TabularSagemakerBackend,
        MultiModalSagemakerBackend,
        TimeSeriesSagemakerBackend,
        TabularRayAWSBackend,
    ]
    __name_to_backend = {cls.name: cls for cls in __supported_backend}

    @staticmethod
    def get_backend(backend: str, **init_args) -> Backend:
        """Return the corresponding backend"""
        assert backend in BackendFactory.__name_to_backend, f"{backend} not supported"
        return BackendFactory.__name_to_backend[backend](**init_args)
