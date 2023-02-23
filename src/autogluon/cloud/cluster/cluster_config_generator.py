from abc import ABC, abstractmethod
from typing import Any, Dict


class ClusterConfigGenerator(ABC):
    @abstractmethod
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get default config of the cluster
        """
        raise NotImplementedError

    @abstractmethod
    def merge_config(new_config) -> Dict[str, Any]:
        """
        Merge specified config with the current one. Settings in new_config will overwrite the old one.
        """
        raise NotImplementedError
