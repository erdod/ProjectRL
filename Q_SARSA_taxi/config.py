import gym
import numpy as np

# Define the environment
env = gym.make("Taxi-v3").env

# Hyperparameters
alpha = 0.1  # learning-rate
gamma = 0.7  # discount-factor
epsilon = 0.1  # explore vs exploit
max_episode_length = 50000  # max episode length
N = 10000  # Total episodes

rng = np.random.default_rng()  # Random generator
