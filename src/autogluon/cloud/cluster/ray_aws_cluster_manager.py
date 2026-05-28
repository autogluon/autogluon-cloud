import logging

from .ray_cluster_manager import RayClusterManager

logger = logging.getLogger(__name__)


class RayAWSClusterManager(RayClusterManager):
    def __init__(self, config: str, cloud_output_bucket: str, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.cloud_output_bucket = cloud_output_bucket
