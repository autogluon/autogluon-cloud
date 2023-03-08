import os
from typing import Any, Dict, Optional, Union

from .cluster_config_generator import DEFAULT_CONFIG_LOCATION
from .constants import (
    ARN,
    AVAILABLE_NODE_TYPES,
    BLOCK_DEVICE_MAPPINGS,
    CLUSTER_NAME,
    DOCKER,
    EBS,
    IAM_INSTANCE_PROFILE,
    IMAGE,
    INSTANCE_TYPE,
    MAX_WORKERS,
    MIN_WORKERS,
    NODE_CONFIG,
    PROVIDER,
    REGION,
    VOLUME_SIZE,
)
from .ray_cluster_config_generator import RayClusterConfigGenerator


class RayAWSClusterConfigGenerator(RayClusterConfigGenerator):
    default_config_file = os.path.join(DEFAULT_CONFIG_LOCATION, "ray_aws_default_cluster_config.yaml")

    def __init__(
        self,
        config: Optional[Union[str, Dict[str, Any]]] = None,
        cluster_name: Optional[str] = "ag_ray_aws_default",
        region: Optional[str] = "us-east-1",
        **kwargs,
    ) -> None:
        """
        Parameter
        ---------
        config, Optional[Union[str, Dict[str, Any]]]
            Config to be used to launch up the cluster. Default: None
            If not set, will use the default config pre-defined.
            If str, must be a path pointing to a yaml file containing the config.
        cluster_name: Optional[str]. Default ag_ray_aws_default
            Name of the cluster being deployed
        region, Optional[str]
            Region to launch the cluster. Default us-east-1
        """
        super().__init__(config=config)
        self._update_cluster_name(cluster_name)
        self._update_region(region)

    def _update_config(
        self,
        instance_type: Optional[str] = None,
        instance_count: Optional[int] = None,
        head_node_name: Optional[str] = "head",
        worker_node_name: Optional[str] = "worker",
        volumes_size: Optional[int] = None,
        custom_image_uri: Optional[str] = None,
        head_instance_profile: Optional[str] = None,
        worker_instance_profile: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Update current config with given parameters.

        Parameters
        ----------
        instance_type: str, default = None
            Instance type the cluster will launch.
            If provided, will overwrite `available_node_types.node_config.InstanceType`
            To learn more,
                https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-configuration.html#node-config
        instance_count: int, default = None
            Number of instance the cluster will launch.
            If provided, will overwrite `available_node_types.min_workers` and `max_workers`
            min_workers and max_workers will both equal to `instance_count` - 1 because there will be a head node.
            This setting doesn't work when there's more than one definition of worker nodes.
            To learn more,
                https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-configuration.html#available-node-types-node-type-name-node-type-min-workers
                https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-configuration.html#max-workers
                https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-configuration.html#cluster-configuration-max-workers
        head_node_name: str, default = head
            Name of the head node inside the yaml file. This is used to correctly configure the head instance profile if specified.
        worker_node_name: str, default = worker
            Name of the worker node inside the yaml file. This is used to correctly configure the instance count and worker instance profile if specified.
        volumes_size: int, default = None
            Size in GB of the EBS volume to use for each of the node.
            If provided, will overwrite `available_node_types.node_config.BlockDeviceMappings.Ebs.VolmueSize`
            To learn more,
                https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-configuration.html#available-node-types
        custom_image_uri: str, default = None
            Custom image to be used by the cluster container. The image MUST have Ray and AG installed.
            If provided, will overwrite `docker.image`
            To learn more,
                https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-configuration.html#docker-image
        head_instance_profile: Optional[str], default = None
            The instance profile arn to be attched to the head node. If not specified, a default one will be created with full EC2 and S3 permissions.
            To learn how to scope down the permission,
                https://github.com/ray-project/ray/issues/9327
        worker_instance_profile: Optional[str], default = None
            The instance profile arn to be attched to the worker node. If not specified, a default one will be created with full EC2 and S3 permissions.
            To learn how to scope down the permission,
                https://github.com/ray-project/ray/issues/9327
        """
        self._update_instance_type(instance_type=instance_type)
        self._update_instance_count(instance_count=instance_count, worker_node_name=worker_node_name)
        self._update_volume_size(volumes_size=volumes_size)
        self._update_custom_image(custom_image_uri=custom_image_uri)
        self._update_instance_profile(node=head_node_name, instance_profile=head_instance_profile)
        self._update_instance_profile(node=worker_node_name, instance_profile=worker_instance_profile)

    def _set_available_node_types(self):
        """Set available node types to be default ones if user didn't provide any"""
        default_config = self._default_config
        available_node_types: Dict[str, Any] = self.config.get(AVAILABLE_NODE_TYPES, None)
        if available_node_types is None:
            available_node_types = default_config[AVAILABLE_NODE_TYPES]
        self.config.update(available_node_types)

    def _set_provider(self):
        """Set provider to be default ones if user didn't provide any"""
        default_config = self._default_config
        provider: Dict[str, Any] = self.config.get(PROVIDER, None)
        if provider is None:
            provider = default_config[PROVIDER]
        self.config.update({PROVIDER: provider})

    def _update_cluster_name(self, cluster_name):
        self.config.update({CLUSTER_NAME: cluster_name})

    def _update_region(self, region):
        self._set_provider()
        self.config[PROVIDER].update({REGION: region})

    def _update_instance_type(self, instance_type):
        if instance_type is not None:
            self._set_available_node_types()
            for node in self.config[AVAILABLE_NODE_TYPES]:
                node_config: Dict[str, Any] = self.config[node].get(NODE_CONFIG, None)
                assert (
                    node_config is not None
                ), f"Detected node definition for {node} but there's no node_config specified. Please provide one."
                self.config[AVAILABLE_NODE_TYPES][node][NODE_CONFIG].update({INSTANCE_TYPE: instance_type})

    def _update_instance_count(self, instance_count, worker_node_name):
        if instance_count is not None:
            worker_instance_count = instance_count - 1
            assert worker_instance_count >= 0
            self.config[MAX_WORKERS] = worker_instance_count
            self._set_available_node_types()
            assert (
                worker_node_name in self.config[AVAILABLE_NODE_TYPES]
            ), f"Didn't find node definition for {worker_node_name}. Please make sure you provided the correct `worker_node_name`"
            self.config[AVAILABLE_NODE_TYPES][worker_node_name].update({MIN_WORKERS: worker_instance_count})

    def _update_volume_size(self, volumes_size):
        if volumes_size is not None:
            self._set_available_node_types()
            for node in self.config[AVAILABLE_NODE_TYPES]:
                node_config: Dict[str, Any] = self.config[node].get(NODE_CONFIG, None)
                assert (
                    node_config is not None
                ), f"Detected node definition for {node} but there's no node_config specified. Please provide one."
                block_mappings = self.config[AVAILABLE_NODE_TYPES][node][NODE_CONFIG][BLOCK_DEVICE_MAPPINGS]
                if BLOCK_DEVICE_MAPPINGS not in node_config:
                    block_mappings = [{"DeviceName": "/dev/sda1", EBS: {VOLUME_SIZE: volumes_size}}]
                else:
                    block_mappings[0][EBS].update({VOLUME_SIZE: volumes_size})

    def _update_custom_image(self, custom_image_uri):
        if custom_image_uri is not None:
            if DOCKER not in self.config:
                self.config[DOCKER] = {}
            self.config[DOCKER].update({IMAGE: custom_image_uri})

    def _update_instance_profile(self, node, instance_profile):
        if instance_profile is not None:
            self._set_available_node_types()
            assert node in self.config[AVAILABLE_NODE_TYPES], f"Specified node {node} is not available in the config."
            node_config: Dict[str, Any] = self.config[node].get(NODE_CONFIG, None)
            assert (
                node_config is not None
            ), f"Detected node definition for {node} but there's no node_config specified. Please provide one."
            self.config[AVAILABLE_NODE_TYPES][node][NODE_CONFIG].update(
                {IAM_INSTANCE_PROFILE: {ARN: instance_profile}}
            )
