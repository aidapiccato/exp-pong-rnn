"""Config."""

import trainer
from agents import random
from envs import exp_pong_env
def get_config():
    """Get config for main.py."""

    config = {
        'constructor': trainer.Trainer,
        'kwargs': {
            'agent': random.Random(),
            'env': exp_pong_env.ExpPongEnv(),    
            'iterations': int(1e6),
            'scalar_eval_every': 100,
            'snapshot_every': 5000,
        },
    }

    return config