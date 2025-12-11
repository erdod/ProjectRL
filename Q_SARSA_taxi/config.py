import gymnasium as gym
import numpy as np

# --- ENVIRONMENT ---
# Usiamo render_mode=None per la velocità durante il training
env = gym.make("Taxi-v3", render_mode=None)

# Generatore di numeri casuali
rng = np.random.default_rng()

# Lunghezza massima di un singolo episodio (sicurezza anti-loop)
max_episode_length = 500 

# --- PARAMETRI Q-LEARNING (Confronto con HRL) ---
# Q-Learning è efficiente, 5000 episodi bastano per battere il gioco.
params_q = {
    "N": 5000,          # Stesso numero del tuo HRL
    "alpha": 0.1,       # Learning rate
    "gamma": 0.99,      # Importante: Taxi richiede visione a lungo termine
    "epsilon": 1.0,     # Start
    "min_epsilon": 0.01,# End
    "decay": 0.001      # Velocità decadimento
}

# --- PARAMETRI SARSA ---
# SARSA è più prudente, diamogli più tempo per convergere.
params_sarsa = {
    "N": 10000,         # Un po' più di episodi
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon": 1.0,
    "min_epsilon": 0.001, # Deve diventare quasi zero alla fine
    "decay": 0.0005
}

# --- PARAMETRI MONTE CARLO ---
# MC ha bisogno di tantissimi dati e nessun discount factor.
params_mc = {
    "N": 20000,         # MC è inefficiente, servono molti episodi
    "gamma": 1.0,       # Nessuno sconto per MC
    "epsilon": 1.0,
    "min_epsilon": 0.01,
    "decay": 0.0002     # Decay molto lento
}