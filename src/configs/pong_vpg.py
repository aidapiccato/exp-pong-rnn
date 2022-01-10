"""Config."""

import trainer
import torch
from agents import vpg
from envs import pong_env
from utils import mlp

def get_policy_net_config():
    config = {
        'constructor': mlp.MLP,
    }
    return config

def get_agent_config():
    config = {
        'constructor': vpg.VPG,
        'kwargs': {
            'policy_net': get_policy_net_config(),
            'optim_config': {
                'optimizer': torch.optim.Adam,
                'kwargs': {
                    'lr': 1e-4,
                },
            },
            'num_actions': 3,
            'discount': 0.9,
            'epsilon': 0.1,
            'batch_size': 8,
            'grad_clip': 1.,
        },       
    }
    return config


def get_config():
    """Get config for main.py."""

    config = {
        'constructor': trainer.Trainer,
        'kwargs': {
            'agent': get_agent_config(),
            'env': pong_env.PongEnv(),    
            'iterations': int(1e10),
            'train_every': 100,
            'image_eval_every': 1000,
            'scalar_eval_every': 100,
            'snapshot_every': 5000,
        },
    }

    return config