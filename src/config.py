import os
import datetime
import torch

from utils import load_flat_options_from_yaml


class Configurable(object):
    """A nested dictionary-like configuration object"""

    DEFAULT_CONFIG = "config-default.yaml"

    def __init__(self, config_path, job_key="", model_key=""):
        """Loads a YAML configuration from the specified path.
        
        :param config_path: path to YAML configuration file
        :param job_key: prefix to all config keys for a job
        :param model_key: prefix to all configuration keys for LM
        """
        self.config_path = config_path

        # Load user-specified options
        self.options = {}
        options = load_flat_options_from_yaml(self.config_path)
        for key, value in options.items():
            self.set_option(key, value)

        # Set default configuration key-value pairs for any unspecified options
        defaults = load_flat_options_from_yaml(Configurable.DEFAULT_CONFIG)
        for key, value in defaults.items():
            if not self.has_option(key):
                self.set_option(key, value)

        self.device = (
            "cuda"
            if self.get_option("job.device") == "cuda" and torch.cuda.is_available()
            else "cpu"
        )
        self.folder = os.path.dirname(self.config_path)
        self.logfile = os.path.join(self.folder, "job.log")
        self.dt_format = "%d/%m/%Y %H:%M:%S"

        self.job_key = job_key
        self.model_key = model_key

    def has_option(self, name):
        """Check if the configuration has the specified option"""
        keys = name.split(".")
        values = self.options
        for key in keys:
            if key not in values:
                return False
            values = values[key]
        return True

    def get_option(self, name, default=None):
        """Get the specified option, with an optional default value"""
        keys = name.split(".")
        values = self.options
        for key in keys:
            if key not in values:
                if default is not None:
                    return default
                raise ValueError(f"key {key} in name {name} not found")
            values = values[key]
        return values

    def set_option(self, name, value):
        """Set an option in the config with a given value"""
        keys = name.split(".")
        values = self.options
        for key in keys[:-1]:
            if key not in values:
                values[key] = {}
            values = values[key]

        values[keys[-1]] = value

    def log(self, msg, echo=True, mode="a"):
        """Log a message"""
        now = datetime.datetime.now().strftime(self.dt_format)
        msg = f"[{now}] {msg}"

        with open(self.logfile, mode) as f:
            f.write(msg + "\n")

        if echo:
            print(msg)

    @staticmethod
    def _flatten(options, result, prefix=""):
        """Credits to https://github.com/uma-pi1/kge/blob/2b693e31c4c06c71336f1c553727419fe01d4aa6/kge/config.py#L383"""
        for key, value in options.items():
            fullkey = key if prefix == "" else prefix + "." + key
            if isinstance(value, dict):
                Configurable._flatten(value, result, prefix=fullkey)
            else:
                result[fullkey] = value
