import gymnasium as gym
import numpy as np

env = gym.make("Taxi-v3", render_mode=None)
rng = np.random.default_rng()
max_episode_length = 200 
test_episodes = 100

params_q = {
    "N": 5000,
    "alpha": 0.1,
    "gamma": 0.99,      
    "epsilon": 1.0,
    "min_epsilon": 0.01,
    "decay": 0.001
}

params_sarsa = {
    "N": 10000,
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon": 1.0,
    "min_epsilon": 0.001, 
    "decay": 0.0005
}

params_mc = {
    "N": 10000,         
    "gamma": 1.0,
    "epsilon": 1.0,
    "min_epsilon": 0.01,
    "decay": 0.0005
}