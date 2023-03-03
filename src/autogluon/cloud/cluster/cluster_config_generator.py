import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import yaml

DEFAULT_CONFIG_LOCATION = os.path.join(os.path.dirname(__file__), "..", "default_cluster_configs")


class ClusterConfigGenerator(ABC):
    default_config_file = os.path.join(DEFAULT_CONFIG_LOCATION, "DEFAULT_CONFIG")

    def __init__(self, config: Optional[Union[str, Dict[str, Any]]] = None, **kwargs) -> None:
        """
        Parameter
        ---------
        config, Optional[Union[str, Dict[str, Any]]]
            Config to be used to launch up the cluster. Default: None
            If not set, will use the default config pre-defined.
            If str, must be a path pointing to a yaml file containing the config.
        """
        if config is None:
            config = self.default_config_file
        if not isinstance(config, dict):
            with open(config, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)
        self.config = config
        self._default_config = self.get_default_config()

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get default config of the cluster
        """
        with open(cls.default_config_file, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config

    def update_config(self, new_config: Union[Dict[str, Any], str] = None, **kwargs) -> Dict[str, Any]:
        """
        Update current config with given one. Settings in new_config will overwrite the old one.

        Parameters
        ----------
        new_config, Optional[Union[Dict[str, Any], str]]
            The new config to overwrite the current config.
            If None, specific keys that needs to be updated must be provided.
            If both new_config and specific arguments to update the config are provided,
            will apply new_config first then override it with each specific aruguments
        """
        if new_config is not None:
            if isinstance(new_config, str):
                with open(new_config, "r") as yaml_file:
                    new_config = yaml.safe_load(yaml_file)
            self.config.update(new_config)
        self._update_config(**kwargs)

        return self.config

    def save_config(self, save_path: str) -> None:
        """
        Save the config to the given path as a yaml file

        Parameter
        ---------
        save_path, str
            Path to save the config. Must be a path pointing to a yaml file
        """
        assert isinstance(
            self.config, dict
        ), f"Invalid config of type: {type(self.config)}. Please provide a dictionary instead"
        with open(save_path, "w") as yaml_file:
            yaml.safe_dump(self.config, yaml_file, default_flow_style=False)

    @abstractmethod
    def _update_config(self, **kwargs) -> None:
        """
        Specific implementations of different cluster config solution
        """
        raise NotImplementedError
