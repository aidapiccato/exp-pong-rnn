"""Config for reloading q-learner from snapshot and generating test figure."""

import tester
import torch
from agents import q_learner
from envs import occ_pong_env
from envs import batch_env

from utils import snapshot

def get_reloaded_params(param_name):
    q_net = snapshot.snapshot(
        '../logs/1',
        235000,
        ['kwargs', 'agent', 'kwargs', param_name],
        freeze_weights=True,
    )
    return q_net

def get_reloaded_q_net():
    return get_reloaded_params('q_net')

def get_reloaded_encoder():
    return get_reloaded_params('encoder')

def get_reloaded_rnn_core():
    return get_reloaded_params('rnn_core')

def get_agent_config():
    config = {
        'constructor': q_learner.QLearner,
        'kwargs': {
            'env': {
                'constructor': batch_env.BatchEnv,
                'kwargs': {
                    'batch_size': 16, 
                    'env_class': occ_pong_env.OccPongEnv,
                    'p_prey': 0.4, 
                    'n_steps': 20,
                    'paddle_radius': 0,
                    'window_width': 10,
                }
            },    
            'q_net': get_reloaded_q_net(),            
            'encoder': get_reloaded_encoder(),
            'rnn_core': get_reloaded_rnn_core(),
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
        'constructor': tester.Tester,
        'kwargs': {
            'agent': get_agent_config(),
            'iterations': 100
        },
    }

    return config