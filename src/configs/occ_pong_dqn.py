"""Config."""

import trainer
import torch
from agents import dqn
from envs import occ_pong
from utils import replay 
from utils import mlp

def get_q_net_config():
    config = {
        'constructor': mlp.MLP,
        'kwargs': {
            'in_features': 10 + 10 + 3, # AIDA: Length of observation (20), plus action vector
            'layer_features': [24, 1],
            'activation': [torch.nn.ReLU(), torch.nn.ReLU(),  torch.nn.ReLU()]
        },
    }
    return config

def get_agent_config():
    config = {
        'constructor': dqn.DQN,
        'kwargs': {
            'q_net': get_q_net_config(),
            'optim_config': {
                'optimizer': torch.optim.Adam,
                'kwargs': {
                    'lr': 1e-4,
                },
            },
            'replay': {
                'constructor': replay.FIFO,
                'kwargs': {
                    # This replay capacity is an important parameter. It has a
                    # lot of influence on learning stability.
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
        'constructor': trainer.Trainer,
        'kwargs': {
            'agent': get_agent_config(),
            'env': occ_pong.OccPongEnv(),    
            'iterations': int(1e10),
            'train_every': 1,
            'image_eval_every': 20*2,
            'scalar_eval_every': 100,
            'snapshot_every': 5000,
        },
    }

    return config