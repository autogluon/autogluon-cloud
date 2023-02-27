from typing import Any, Dict, Union

from cluster_config_generator import ClusterConfigGenerator


class RayClusterConfigGenerator(ClusterConfigGenerator):
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
