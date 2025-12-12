import gymnasium as gym
import numpy as np

# --- ENVIRONMENT ---
# We use render_mode=None for speed during training
env = gym.make("Taxi-v3", render_mode=None)
rng = np.random.default_rng()

# Reduced step limit to speed up Monte Carlo
max_episode_length = 200 

# Number of episodes to run for the TESTING phase
test_episodes = 100

# --- Q-LEARNING PARAMETERS ---
params_q = {
    "N": 5000,
    "alpha": 0.1,
    "gamma": 0.99,      
    "epsilon": 1.0,
    "min_epsilon": 0.01,
    "decay": 0.001
}

# --- SARSA PARAMETERS ---
params_sarsa = {
    "N": 10000,
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon": 1.0,
    "min_epsilon": 0.001, 
    "decay": 0.0005
}

# --- MONTE CARLO PARAMETERS ---
params_mc = {
    "N": 10000,         
    "gamma": 1.0,
    "epsilon": 1.0,
    "min_epsilon": 0.01,
    "decay": 0.0005
}