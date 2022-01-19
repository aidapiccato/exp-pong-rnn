"""Config."""

import tester
import torch
from agents import dqn
from envs import exp_env
from utils import snapshot
from utils import replay
def get_reloaded_q_net():
    q_net = snapshot.snapshot(
        '../logs/2',
        995000,
        ['kwargs', 'agent', 'kwargs', 'q_net'],
        freeze_weights=True,
    )
    return q_net

def get_agent_config():
    config = {
        'constructor': dqn.DQN,
        'kwargs': {
    'q_net': get_reloaded_q_net(),
            'optim_config': {
                'optimizer': torch.optim.Adam,
                'kwargs': {
                    'lr': 1e-4,
                },
            },
            'replay': {
                'constructor': replay.FIFO,
                'kwargs': {
                    'capacity': int(1e5),
                },
            },
            'num_actions': 3,
            'discount': 0.6,
            'epsilon': 0.5,
            'batch_size': 10,
            'grad_clip': 1.,
        },       
    }
    return config



def get_config():
    """Get config for main.py."""

    config = {
        'constructor': tester.Tester,
        'kwargs': {
            'agent': get_agent_config(),
            'env': exp_env.ExpEnv(),    
            'iterations': int(1e4),
        },
    }

    return config