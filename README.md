# exp_pong_rnn

RNNs for the occluded pong task. 

# How to run
Run python main.py -config=<config-name> -log_directory=<log-directory> to run locally. 

# Configs
- occ_q_learner: Q-Learner config
- occ_q_learner_test: Q-Learner config for generating summary performance images

# Directories
- agents: RL Agents
    - q_learner: Q-Learner agent
- configs: Configs of models and experiments
- utils: Additional code files
- envs: RL Environmments
    - batch_env: Wrapper for training over batches
    - occ_pong_env: Occluded pong environment