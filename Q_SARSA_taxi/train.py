import numpy as np
import gymnasium as gym
# Importiamo l'ambiente e i dizionari dei parametri
from config import env, max_episode_length, rng, params_q, params_sarsa, params_mc

# --- Helper Function ---
def get_valid_action(state, env):
    """
    Funzione di aiuto per gestire action_mask su Gymnasium.
    Utile per Monte Carlo per evitare troppe azioni illegali all'inizio.
    """
    try:
        action_mask_fn = env.get_wrapper_attr('action_mask')
        mask = action_mask_fn(state)
        return env.action_space.sample(mask)
    except AttributeError:
        return env.action_space.sample()

# --- Q-LEARNING ---
def train_q_learning():
    print("Q-Learning Started")
    
    # Unpacking dei parametri dal config
    N = params_q["N"]
    alpha = params_q["alpha"]
    gamma = params_q["gamma"]
    epsilon = params_q["epsilon"]
    min_epsilon = params_q["min_epsilon"]
    decay_rate = params_q["decay"]

    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards_arr = []
    num_episodes_arr = []
    
    for i in range(1, N + 1):
        rewards = []
        state, info = env.reset()
        done = False
        lgt = 0
        
        while not done:
            if lgt > max_episode_length:
                break
            
            # Epsilon-Greedy
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update
            current_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            q_table[state, action] = (1 - alpha) * current_value + alpha * (reward + gamma * next_max)
            
            rewards.append(reward)
            state = next_state
            lgt += 1
        
        # Decay
        epsilon = max(min_epsilon, epsilon * np.exp(-decay_rate))
        
        if len(rewards) > 0:
            rewards_arr.append(sum(rewards) / len(rewards))
        else:
            rewards_arr.append(0)
        num_episodes_arr.append(lgt)

    print(f"Q-Learning Avg Reward (last 100): {np.mean(rewards_arr[-100:]):.2f}")
    print("Q-Learning Ended")
    return rewards_arr, num_episodes_arr

# --- SARSA ---
def train_sarsa():
    print("SARSA Started")
    
    # Unpacking parametri
    N = params_sarsa["N"]
    alpha = params_sarsa["alpha"]
    gamma = params_sarsa["gamma"]
    epsilon = params_sarsa["epsilon"]
    min_epsilon = params_sarsa["min_epsilon"]
    decay_rate = params_sarsa["decay"]

    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards_arr = []
    num_episodes_arr = []
    
    for i in range(1, N + 1):
        state, info = env.reset()
        rewards = []
        done = False
        lgt = 0
        
        # Prima azione
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        while not done:
            if lgt > max_episode_length:
                break

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Prossima azione (On-Policy)
            if rng.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[next_state])
            
            # Update
            current_value = q_table[state, action]
            next_val = q_table[next_state, next_action]
            q_table[state, action] = (1 - alpha) * current_value + alpha * (reward + gamma * next_val)
            
            rewards.append(reward)
            state = next_state
            action = next_action
            lgt += 1
            
        epsilon = max(min_epsilon, epsilon * np.exp(-decay_rate))
        
        if len(rewards) > 0:
            rewards_arr.append(sum(rewards) / len(rewards))
        else:
            rewards_arr.append(0)
        num_episodes_arr.append(lgt)

    print(f"SARSA Avg Reward (last 100): {np.mean(rewards_arr[-100:]):.2f}")
    print("SARSA ended")
    return rewards_arr, num_episodes_arr

# --- MONTE CARLO ---
def train_mc():
    print("MC Started")
    
    # Unpacking parametri
    N = params_mc["N"]
    gamma = params_mc["gamma"]
    epsilon = params_mc["epsilon"]
    min_epsilon = params_mc["min_epsilon"]
    decay_rate = params_mc["decay"]

    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    returns_sum = np.zeros([env.observation_space.n, env.action_space.n])
    returns_count = np.zeros([env.observation_space.n, env.action_space.n])

    rewards_arr = []
    num_episodes_arr = []
    
    for k in range(1, N + 1):
        state, info = env.reset()
        episode = [] 
        done = False
        lgt = 0
        
        while not done:
            if lgt > max_episode_length:
                break
            
            # Scelta azione
            if rng.random() < epsilon:
                action = get_valid_action(state, env)
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode.append((state, action, reward))
            state = next_state
            lgt += 1
        
        epsilon = max(min_epsilon, epsilon * np.exp(-decay_rate))
        num_episodes_arr.append(lgt)
        
        ep_rewards = [x[2] for x in episode]
        if len(ep_rewards) > 0:
            rewards_arr.append(sum(ep_rewards) / len(ep_rewards))
        else:
            rewards_arr.append(0)

        # Update Q-Table (Every-Visit)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            s_t, a_t, r_t = episode[t]
            G = gamma * G + r_t
            
            returns_count[s_t, a_t] += 1
            returns_sum[s_t, a_t] += G
            q_table[s_t, a_t] = returns_sum[s_t, a_t] / returns_count[s_t, a_t]

    print(f"MC Avg Reward (last 100): {np.mean(rewards_arr[-100:]):.2f}")
    print("MC Ended")
    return rewards_arr, num_episodes_arr