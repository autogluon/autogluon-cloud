import os

from .cluster_config_generator import DEFAULT_CONFIG_LOCATION, ClusterConfigGenerator


class RayClusterConfigGenerator(ClusterConfigGenerator):
    default_config_file = os.path.join(DEFAULT_CONFIG_LOCATION, "RAY_DUMMY")
