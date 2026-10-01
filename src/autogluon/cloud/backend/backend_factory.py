from .backend import Backend
from .multimodal_sagemaker_backend import MultiModalSagemakerBackend
from .sagemaker_backend import SagemakerBackend
from .tabular_sagemaker_backend import TabularSagemakerBackend
from .timeseries_sagemaker_backend import TimeSeriesSagemakerBackend


class BackendFactory:
    _BACKENDS = {
        SagemakerBackend.name: SagemakerBackend,
        TabularSagemakerBackend.name: TabularSagemakerBackend,
        MultiModalSagemakerBackend.name: MultiModalSagemakerBackend,
        TimeSeriesSagemakerBackend.name: TimeSeriesSagemakerBackend,
    }

    @staticmethod
    def get_backend_cls(backend: str) -> type[Backend]:
        if backend in BackendFactory._BACKENDS:
            return BackendFactory._BACKENDS[backend]
        raise ValueError(f"{backend} not supported. Supported backends: {sorted(BackendFactory._BACKENDS)}")

    @staticmethod
    def get_backend(backend: str, **init_args) -> Backend:
        """Return the corresponding backend"""
        return BackendFactory.get_backend_cls(backend)(**init_args)
