from .backend import Backend
from .sagemaker_backend import SagemakerBackend


class BackendFactory:
    __supported_backend = [SagemakerBackend]
    __backend_name_to_backend = {cls().name: cls for cls in __supported_backend}

    @staticmethod
    def get_backend(backend_name: str) -> Backend:
        """Return the corresponding converter"""
        assert backend_name in BackendFactory.__backend_name_to_backend, f"{backend_name} not supported"
        return BackendFactory.__backend_name_to_backend[backend_name]()
