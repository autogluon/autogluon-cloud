from abc import ABC, abstractmethod
from typing import Dict

from .cluster_config_generator import ClusterConfigGenerator


class ClusterManager(ABC):
    def __init__(self) -> None:
        self.cluster_config_generator = ClusterConfigGenerator()

    @abstractmethod
    @staticmethod
    def generate_default_permission(self, **kwargs) -> Dict[str, str]:
        """Generate default permission file user could use to setup the corresponding entity, i.e. IAM Role in AWS"""
        raise NotImplementedError

    @abstractmethod
    def up(self, config) -> None:
        """
        Launch up the cluster
        """
        raise NotImplementedError

    @abstractmethod
    def down(self) -> None:
        """
        Tear down the cluster
        """
        raise NotImplementedError

    @abstractmethod
    def configure_ray_on_cluster(self) -> None:
        """
        Configure ray runtime on the cluster if not a ray cluster already, i.e. k8s
        """
        raise NotImplementedError

    @abstractmethod
    def setup_connection(self, port) -> None:
        """
        Setup connection between local and remote ray cluster to enable job submission
        """
        raise NotImplementedError

    @abstractmethod
    def setup_dashboard(self, port) -> None:
        """
        Setup ray dashboard to monitor cluster status
        """
        raise NotImplementedError

    @abstractmethod
    def exec(self, command) -> None:
        """
        Execute the command on the head node of the cluster
        """
        raise NotImplementedError
