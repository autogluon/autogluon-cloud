from .backend import Backend
from .constant import RAY_AWS, TABULAR_RAY_AWS
from .multimodal_sagemaker_backend import MultiModalSagemakerBackend
from .sagemaker_backend import SagemakerBackend
from .tabular_sagemaker_backend import TabularSagemakerBackend
from .timeseries_sagemaker_backend import TimeSeriesSagemakerBackend


class BackendFactory:
    _SAGEMAKER_BACKENDS = {
        SagemakerBackend.name: SagemakerBackend,
        TabularSagemakerBackend.name: TabularSagemakerBackend,
        MultiModalSagemakerBackend.name: MultiModalSagemakerBackend,
        TimeSeriesSagemakerBackend.name: TimeSeriesSagemakerBackend,
    }
    _RAY_BACKEND_NAMES = {RAY_AWS, TABULAR_RAY_AWS}

    @staticmethod
    def get_backend_cls(backend: str) -> type[Backend]:
        if backend in BackendFactory._SAGEMAKER_BACKENDS:
            return BackendFactory._SAGEMAKER_BACKENDS[backend]
        if backend in BackendFactory._RAY_BACKEND_NAMES:
            try:
                from .ray_aws_backend import RayAWSBackend, TabularRayAWSBackend
            except ImportError as e:
                raise ImportError(
                    f"The '{backend}' backend requires ray. Install with: pip install 'autogluon.cloud[ray]'"
                ) from e
            return {
                RayAWSBackend.name: RayAWSBackend,
                TabularRayAWSBackend.name: TabularRayAWSBackend,
            }[backend]
        supported = sorted(BackendFactory._SAGEMAKER_BACKENDS.keys() | BackendFactory._RAY_BACKEND_NAMES)
        raise ValueError(f"{backend} not supported. Supported backends: {supported}")

    @staticmethod
    def get_backend(backend: str, **init_args) -> Backend:
        """Return the corresponding backend"""
        return BackendFactory.get_backend_cls(backend)(**init_args)
