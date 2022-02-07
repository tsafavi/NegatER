import yaml
import torch


def load_options_from_yaml(fname):
    """Load a YAML options dictionary from a specified file"""
    with open(fname, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def load_flat_options_from_yaml(fname):
    """Load YAML options dictionary from file with hierarchical keys flattened"""
    options = load_options_from_yaml(fname)
    return flatten_options(options)


def flatten_options(options):
    """Flatten a hierarchical dictionary"""
    result = {}
    _flatten(options, result)
    return result


def save_options_to_yaml(options, fname):
    """Save a dictionary of options to a YAML file"""
    with open(fname, "w") as f:
        yaml.dump(options, f)


def save_checkpoint(save_dict, checkpoint_name):
    """Save a dictionary under a given filename"""
    torch.save(save_dict, checkpoint_name)


def load_checkpoint(checkpoint_name, device="cuda"):
    """Load a checkpoint with pytorch"""
    return torch.load(checkpoint_name, map_location=device)


def _flatten(options, result, prefix=""):
    """Credits to https://github.com/uma-pi1/kge/blob/2b693e31c4c06c71336f1c553727419fe01d4aa6/kge/config.py#L383"""
    for key, value in options.items():
        fullkey = key if prefix == "" else prefix + "." + key
        if type(value) is dict:
            _flatten(value, result, prefix=fullkey)
        else:
            result[fullkey] = value
