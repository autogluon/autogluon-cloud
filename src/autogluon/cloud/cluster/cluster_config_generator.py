from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import os

DEFAULT_CONFIG_LOCATION = os.path.join(__file__, "..", "default_cluster_configs")

class ClusterConfigGenerator(ABC):
    default_config = os.path.join(DEFAULT_CONFIG_LOCATION, "DUMMY")

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get default config of the cluster
        """
        return cls.default_config

    @abstractmethod
    def update_config(new_config: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Update current config with given one. Settings in new_config will overwrite the old one.
        """
        raise NotImplementedError
