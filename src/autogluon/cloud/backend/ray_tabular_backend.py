from __future__ import annotations

import logging
from typing import Any, Dict

import boto3

from ..cluster.ray_aws_cluster_config_generator import RayAWSClusterConfigGenerator
from ..cluster.ray_aws_cluster_manager import RayAWSClusterManager
from ..cluster.ray_cluster_config_generator import RayClusterConfigGenerator
from ..cluster.ray_cluster_manager import RayClusterManager
from .constant import TABULAR_RAY
from .ray_backend import RayBackend

logger = logging.getLogger(__name__)


class RayTabularBackend(RayBackend):
    name = TABULAR_RAY

    @property
    def _cluster_config_generator(self) -> RayClusterConfigGenerator:
        return RayAWSClusterConfigGenerator

    @property
    def _cluster_manager(self) -> RayClusterManager:
        return RayAWSClusterManager

    @property
    def _config_file_name(self) -> str:
        return "ag_ray_aws_cluster_config.yaml"

    def initialize(self, **kwargs) -> None:
        """Initialize the backend."""
        super().initialize(**kwargs)
        self._boto_session = boto3.session.Session()
        self.region = self._boto_session.region_name
        assert (
            self.region is not None
        ), "Please setup a region via `export AWS_DEFAULT_REGION=YOUR_REGION` in the terminal"

    def generate_default_permission(self, **kwargs) -> Dict[str, str]:
        """Generate default permission file user could use to setup the corresponding entity, i.e. IAM Role in AWS"""
        return RayAWSClusterManager.generate_default_permission(**kwargs)

    def parse_backend_fit_kwargs(self, kwargs: Dict) -> Dict[str, Any]:
        """Parse backend specific kwargs and get them ready to be sent to fit call"""
        raise NotImplementedError
