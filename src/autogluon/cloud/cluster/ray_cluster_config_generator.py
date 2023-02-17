from typing import Any, Dict

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

    def merge_config(new_config) -> Dict[str, Any]:
        """
        Merge specified config with the current one. Settings in new_config will overwrite the old one.
        """
        raise NotImplementedError
