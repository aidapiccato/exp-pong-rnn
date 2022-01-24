"""Config."""

import episode_trainer
import torch
from agents import q_learner
from envs import occ_pong_env
from envs import batch_env

from utils import mlp

_HIDDEN_SIZE = 256
_ENC_SIZE = 256
_NUM_ACTIONS = 3
_OBS_SIZE = 2 * 10


def get_q_net_config():
    config = {
        'constructor': mlp.MLP,
        'kwargs': {
            'in_features': _HIDDEN_SIZE + _NUM_ACTIONS,
            'layer_features': [256, 1],
        },
    }
    return config


def get_encoder_config():
    config = {
        'constructor': mlp.MLP,
        'kwargs': {
            'in_features': _OBS_SIZE,
            'layer_features': [256, _ENC_SIZE],
            'activate_final': True,
        },
    }
    return config


def get_rnn_core_config():
    config = {
        'constructor': mlp.MLP,
        'kwargs': {
            'in_features': _ENC_SIZE + _HIDDEN_SIZE,
            'layer_features': [256, _HIDDEN_SIZE],
            'activate_final': True,
        },
    }
    return config


def get_agent_config():
    config = {
        'constructor': q_learner.QLearner,
        'kwargs': {
            'env': {
                'constructor': batch_env.BatchEnv,
                'kwargs': {
                    'batch_size': 16, 
                    'env_class': occ_pong_env.OccPongEnv,
                    'p_prey': 0.9, 
                    'n_steps': 20,
                    'paddle_radius': 1,
                    'window_width': 10,
                }
            },    
            'q_net': get_q_net_config(),            
            'encoder': get_encoder_config(),
            'rnn_core': get_rnn_core_config(),
            'optim_config': {
                'optimizer': torch.optim.Adam,
                'kwargs': {
                    'lr': 1e-4,
                },
            },
            'discount': 0.1,
            'epsilon': 1.,
        },       
    }
    return config



def get_config():
    """Get config for main.py."""

    config = {
        'constructor': episode_trainer.EpisodeTrainer,
        'kwargs': {
            'agent': get_agent_config(),
            'episodes': int(1e6),
            'image_eval_every': 20*2,
            'scalar_eval_every': 100,
            'snapshot_every': 5000,
        },
    }

    return config