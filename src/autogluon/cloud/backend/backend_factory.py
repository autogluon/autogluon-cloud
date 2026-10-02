from typing import Optional, Union

from ..config import SageMakerConfig
from .backend import Backend
from .multimodal_sagemaker_backend import MultiModalSagemakerBackend
from .sagemaker_backend import SagemakerBackend
from .tabular_sagemaker_backend import TabularSagemakerBackend
from .timeseries_sagemaker_backend import TimeSeriesSagemakerBackend


class BackendFactory:
    _CONFIGS = {SageMakerConfig.name: SageMakerConfig}
    _BACKENDS = {
        SagemakerBackend.name: SagemakerBackend,
        TabularSagemakerBackend.name: TabularSagemakerBackend,
        MultiModalSagemakerBackend.name: MultiModalSagemakerBackend,
        TimeSeriesSagemakerBackend.name: TimeSeriesSagemakerBackend,
    }

    @staticmethod
    def resolve_config(backend: Union[str, SageMakerConfig]) -> SageMakerConfig:
        """Normalize a backend name or reusable configuration without creating resources."""
        if isinstance(backend, SageMakerConfig):
            return backend
        if not isinstance(backend, str):
            raise TypeError("`backend` must be a backend name or SageMakerConfig.")
        if backend in ("ray", "ray_aws"):
            raise ValueError("The Ray backend was removed in AutoGluon-Cloud v0.7.0. Use backend='sagemaker' instead.")
        if backend not in BackendFactory._CONFIGS:
            raise ValueError(
                f"Unsupported backend {backend!r}. Supported backends: {sorted(BackendFactory._CONFIGS)}."
            )
        return BackendFactory._CONFIGS[backend]()

    @staticmethod
    def get_backend_cls(backend: str) -> type[Backend]:
        if backend in BackendFactory._BACKENDS:
            return BackendFactory._BACKENDS[backend]
        raise ValueError(f"{backend} not supported. Supported backends: {sorted(BackendFactory._BACKENDS)}")

    @staticmethod
    def get_backend(
        backend: str,
        *,
        local_output_path: str,
        cloud_output_path: Optional[str],
        predictor_type: str,
        config: Optional[SageMakerConfig] = None,
        resource_prefix: Optional[str] = None,
    ) -> Backend:
        """Create a backend with its own execution state from reusable settings."""
        return BackendFactory.get_backend_cls(backend)(
            local_output_path=local_output_path,
            cloud_output_path=cloud_output_path,
            predictor_type=predictor_type,
            config=config,
            resource_prefix=resource_prefix,
        )
