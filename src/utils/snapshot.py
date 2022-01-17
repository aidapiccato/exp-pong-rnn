"""Method to get model restored from snapshot."""

import importlib
import os
from python_utils.configs import build_from_config
from python_utils.configs import override_config
import numpy as np
import torch


def snapshot(log_dir,
             snapshot_num,
             model_node,
             freeze_weights=False,
             config_overrides=''):
    """Get a model restored from a snapshot.
    
    Args:
        log_dir: String. Path to log directory for model to restore.
        snapshot_num: Int. Snapshot number. Must be the name of a file in
            $log_dir/snapshots.
        model_node: Iterable of strings. Path to model node in config.
        freeze_weights: Bool. Whether to prevent gradient updates of the weights
            of the model.
        config_overrides: String. JSON-serialized dictionary of config overrides
            to impose in addition to those used when the model was trained.
            
    Returns:
        model: Constructed model with parameters restored from snapshot.
    """
    snapshot_path = os.path.join(log_dir, 'snapshots', str(snapshot_num))
    config_name = open(os.path.join(log_dir, 'config_name.log'), 'r').read()
    
    # Load config
    config_module = importlib.import_module(config_name)
    config = config_module.get_config()
    
    # Find and apply config overrides used when training the snapshot
    f_config_overrides = os.path.join(log_dir, 'config_overrides.log')
    f_config_overrides_open = open(f_config_overrides, 'r')
    s_config_overrides = f_config_overrides_open.read()
    # Must replace single quotes from json dump by double quotes, because
    # double quotes were converted to single quotes in
    # generate_config_overrides.py.
    s_config_overrides = s_config_overrides.replace("'", '"')
    config = override_config.override_config_from_json(
        config, s_config_overrides)
        
    # Now also apply the additional config_overrides passed in as an argument
    config = override_config.override_config(
        config, config_overrides)
    
    # Build model
    for k in model_node:
        config = config[k]
    model = build_from_config.build_from_config(config)
    
    # Restore model parameters from snapshot
    model.load_state_dict(torch.load(snapshot_path))
    
    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False
            
    return model