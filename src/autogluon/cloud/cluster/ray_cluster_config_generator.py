from typing import Any, Dict, Union

from .cluster_config_generator import ClusterConfigGenerator, DEFAULT_CONFIG_LOCATION

import os


class RayClusterConfigGenerator(ClusterConfigGenerator):
    default_config = os.path.join(DEFAULT_CONFIG_LOCATION, "ray_default_cluster_config.yaml")
    
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get default config of the cluster
        """
        raise NotImplementedError

    def update_config(new_config: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Update current config with given one. Settings in new_config will overwrite the old one.
        """
        raise NotImplementedError
