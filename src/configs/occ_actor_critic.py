"""Config."""

import episode_trainer
from agents import actor_critic
from envs import occ_pong_env
import torch


def get_actor_net():
    pass

def get_critic_net():
    pass

def get_state_net():
    pass

def get_agent_config():
    config = {
        'constructor': actor_critic.ActorCritic,
        'kwargs': {
            'actor_net': get_actor_net(),
            'critic_net': get_critic_net(),
            'state_net': get_state_net(),
            'num_actions': 3, 
            'discount': 0.9, 
            'optim_config': {
                'actor_optimizer': torch.optim.Adam,
                'actor_kwargs': {
                    'lr': 1e-4,
                },
                'critic_optimizer': torch.optim.Adam,
                'critic_kwargs': {
                    'lr': 1e-4,
                },
            }
        }

    }
    return config

def get_config():
    """Get config for main.py."""

    config = {
        'constructor': episode_trainer.EpisodeTrainer,
        'kwargs': {
            'agent': get_agent_config(),
            'batch_size': 2, 
            'env_class': occ_pong_env.OccPongEnv,    
            'env_kwargs': {
                'p_prey': 0.2,
                'max_t': 30
            },
            'episodes': int(1e10),
            'image_eval_every': 20*2,
            'scalar_eval_every': 100,
            'snapshot_every': 5000,
        },
    }

    return config