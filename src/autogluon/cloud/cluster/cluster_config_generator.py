from abc import ABC, abstractmethod
from typing import Any, Dict, Union


class ClusterConfigGenerator(ABC):
    @abstractmethod
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get default config of the cluster
        """
        raise NotImplementedError

    @abstractmethod
    def update_config(new_config: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Update current config with given one. Settings in new_config will overwrite the old one.
        """
        raise NotImplementedError
