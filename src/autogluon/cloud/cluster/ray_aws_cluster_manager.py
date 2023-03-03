from typing import Dict

from .ray_cluster_manager import RayClusterManager


class RayAWSClusterManager(RayClusterManager):
    @staticmethod
    def generate_default_permission() -> Dict[str, str]:
        """
        Generate trust relationship and iam policy required to manage cluster
        """
        raise NotImplementedError

    def setup_connection(self, port) -> None:
        """
        Setup connection between local and remote ray cluster to enable job submission
        """
        raise NotImplementedError

    def setup_dashboard(self, port) -> None:
        """
        Setup ray dashboard to monitor cluster status
        """
        raise NotImplementedError

    def exec(self, command) -> None:
        """
        Execute the command on the head node of the cluster
        """
        raise NotImplementedError
