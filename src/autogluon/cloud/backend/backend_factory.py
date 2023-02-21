from .backend import Backend
from .sagemaker_backend import SagemakerBackend
from .timeseries_sagemaker_backend import TimeSeriesSagemakerBackend


class BackendFactory:
    __supported_backend = [SagemakerBackend, TimeSeriesSagemakerBackend]
    __name_to_backend = {cls().ext: cls for cls in __supported_backend}

    @staticmethod
    def get_backend(backend: str) -> Backend:
        """Return the corresponding converter"""
        assert backend in BackendFactory.__name_to_backend, f"{backend} not supported"
        return BackendFactory.__name_to_backend[backend]()
