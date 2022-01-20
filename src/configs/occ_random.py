"""Config."""

import batch_trainer
from agents import random
from envs import occ_pong


def get_config():
    """Get config for main.py."""

    config = {
        'constructor': batch_trainer.BatchTrainer,
        'kwargs': {
            'agent': random.Random(),
            'batch_size': 2, 
            'env_class': occ_pong.OccPongEnv,    
            'env_kwargs': {},
            'iterations': int(1e10),
            'train_every': 1,
            'image_eval_every': 20*2,
            'scalar_eval_every': 100,
            'snapshot_every': 5000,
        },
    }

    return config