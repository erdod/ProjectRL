import numpy as np
from config import env, rng, epsilon

def choose_action(state, q_table):
    """
    Choose an action using an epsilon-greedy policy.
    """
    if rng.random() < epsilon:
        action = env.action_space.sample()  # Explore the action space
    else:
        action = np.argmax(q_table[state])  # Exploit learned values
    return action

def save_model(q_table, filename="q_table.npy"):
    """
    Saves the Q-table to a file.
    """
    np.save(filename, q_table)
    print(f"Model saved to {filename}")

def load_model(filename="q_table.npy"):
    """
    Loads the Q-table from a file.
    """
    print(f"Loading model from {filename}")
    return np.load(filename)
