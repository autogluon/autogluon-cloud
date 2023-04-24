import logging
import os
import subprocess
from typing import Dict, List, Optional

from .cluster_manager import ClusterManager

logger = logging.getLogger(__name__)


class RayClusterManager(ClusterManager):
    @staticmethod
    def generate_default_permission() -> Dict[str, str]:
        """
        Generate default permission required to manage cluster
        """
        raise NotImplementedError

    def up(self, config: Optional[str] = None, ray_up_args: Optional[List[str]] = None, **kwargs) -> None:
        """
        Launch up the cluster with a given config

        Parameter
        ---------
        config, Optional[str]
            Path to a yaml file defining the configuration of the cluster.
            If not provided, will use the config provided when initialized.
        ray_up_args: Optional[List[str]]
            Additional ray up arguments to be passed.
            To learn more,
                https://docs.ray.io/en/latest/cluster/cli.html#ray-up
        """
        if ray_up_args is None:
            ray_up_args = []
        if config is None:
            config = self.config
        cmd = ["ray", "up", config, "-y", "--disable-usage-stats", "--no-config-cache"] + ray_up_args
        result = subprocess.run(cmd, check=True)

        if result.returncode == 0:
            self.config = config

    def down(self, ray_down_args: Optional[List[str]] = None, **kwargs) -> None:
        """
        Tear down the cluster

        Parameter
        ---------
        ray_down_args: Optional[List[str]]
            Additional ray up arguments to be passed.
            To learn more,
                https://docs.ray.io/en/latest/cluster/cli.html#ray-down
        """
        if ray_down_args is None:
            ray_down_args = []
        cmd = ["ray", "down", self.config, "-y"] + ray_down_args
        subprocess.run(cmd, check=True)

    def configure_ray_on_cluster(self) -> None:
        """
        Configure ray runtime on the cluster if not a ray cluster already
        """
        # ray runtime will be automatically configured with ray up

    def setup_connection(self, port: int = 8265, **kwargs) -> None:
        """
        Setup connection between local and remote ray cluster to enable job submission.
        This method calls `setup_dashboard()` with non-blocking flag underneath to make the connection

        Parameters
        ----------
        port, int
            The local port to make the connection
        """
        self.setup_dashboard(port=port, block=False)
        os.environ["RAY_ADDRESS"] = f"http://127.0.0.1:{port}"
        logger.log(20, "Connection was set successfully. Ready to submit jobs.")

    def setup_dashboard(self, port: int = 8265, block: bool = False, **kwargs) -> None:
        """
        Setup ray dashboard to monitor cluster status

        Parameters
        ----------
        port, int
            The local port to forward to the dashboard
        block, bool
            Whether to make this a blocking call or not
        """
        cmd = f"ray dashboard -p {port} {self.config}"
        if not block:
            cmd = "nohup " + cmd + " >/dev/null 2>&1 &"
        result = subprocess.run(cmd, shell=True, check=True)
        if result.returncode != 0:
            error_msg = "Failed to setup the dashboard."
            if block:
                error_msg += " Please check the failure reason in file `nohup.out`"
            raise ValueError(error_msg)
        logger.log(20, f"Dashboard is available at localhost:{port}")

    def exec(
        self,
        command,
        ray_exec_args: Optional[List[str]] = None,
    ) -> None:
        """
        Execute the command on the head node of the cluster
        """
        if ray_exec_args is None:
            ray_exec_args = []
        cmd = ["ray", "exec", *ray_exec_args, self.config, command]
        subprocess.run(cmd, check=True)
