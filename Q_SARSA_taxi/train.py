import numpy as np
import gymnasium as gym
from config import env, max_episode_length, rng, params_q, params_sarsa, params_mc

# --- Helper Function ---
def get_valid_action(state, env):
    try:
        action_mask_fn = env.get_wrapper_attr('action_mask')
        mask = action_mask_fn(state)
        return env.action_space.sample(mask)
    except AttributeError:
        return env.action_space.sample()

# --- EVALUATION FUNCTION (TESTING) ---
def evaluate_agent(q_table, episodes, name="Agent"):
    """
    Evaluates the trained agent for a number of episodes.
    Returns: rewards array, steps array, success count
    """
    print(f"\n--- Testing {name} for {episodes} episodes ---")
    total_rewards = []
    total_steps = []
    success_count = 0
    
    for _ in range(episodes):
        state, info = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        
        while not done:
            # Exploitation only (Greedy)
            action = np.argmax(q_table[state])
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            # Check for cutoff (usually not needed in test if optimized, but good for safety)
            if steps > max_episode_length:
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        # In Taxi, 'terminated' means success (drop off), 'truncated' means timeout
        if terminated:
            success_count += 1
            
    print(f"Success Rate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
    return total_rewards, total_steps, success_count

# --- Q-LEARNING ---
def train_q_learning():
    print("\n--- Q-Learning Started ---")
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
            if lgt > max_episode_length: break
            
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            current_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            q_table[state, action] = (1 - alpha) * current_value + alpha * (reward + gamma * next_max)
            
            rewards.append(reward)
            state = next_state
            lgt += 1
        
        epsilon = max(min_epsilon, epsilon * np.exp(-decay_rate))
        
        if len(rewards) > 0: rewards_arr.append(sum(rewards) / len(rewards))
        else: rewards_arr.append(0)
        num_episodes_arr.append(lgt)

    print(f"Q-Learning Ended. Avg Reward (last 100 eps): {np.mean(rewards_arr[-100:]):.2f}")
    # RETURN Q_TABLE TOO
    return rewards_arr, num_episodes_arr, q_table

# --- SARSA ---
def train_sarsa():
    print("\n--- SARSA Started ---")
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
        
        if rng.random() < epsilon: action = env.action_space.sample()
        else: action = np.argmax(q_table[state])
        
        while not done:
            if lgt > max_episode_length: break

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if rng.random() < epsilon: next_action = env.action_space.sample()
            else: next_action = np.argmax(q_table[next_state])
            
            current_value = q_table[state, action]
            next_val = q_table[next_state, next_action]
            q_table[state, action] = (1 - alpha) * current_value + alpha * (reward + gamma * next_val)
            
            rewards.append(reward)
            state = next_state
            action = next_action
            lgt += 1
            
        epsilon = max(min_epsilon, epsilon * np.exp(-decay_rate))
        
        if len(rewards) > 0: rewards_arr.append(sum(rewards) / len(rewards))
        else: rewards_arr.append(0)
        num_episodes_arr.append(lgt)

    print(f"SARSA Ended. Avg Reward (last 100 eps): {np.mean(rewards_arr[-100:]):.2f}")
    # RETURN Q_TABLE TOO
    return rewards_arr, num_episodes_arr, q_table

# --- MONTE CARLO ---
def train_mc():
    print("\n--- MC Started ---")
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
            if lgt > max_episode_length: break
            
            if rng.random() < epsilon: action = env.action_space.sample()
            else: action = np.argmax(q_table[state])
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode.append((state, action, reward))
            state = next_state
            lgt += 1
        
        epsilon = max(min_epsilon, epsilon * np.exp(-decay_rate))
        num_episodes_arr.append(lgt)
        
        ep_rewards = [x[2] for x in episode]
        if len(ep_rewards) > 0: rewards_arr.append(sum(ep_rewards) / len(ep_rewards))
        else: rewards_arr.append(0)

        G = 0
        for t in range(len(episode) - 1, -1, -1):
            s_t, a_t, r_t = episode[t]
            G = gamma * G + r_t
            returns_count[s_t, a_t] += 1
            returns_sum[s_t, a_t] += G
            q_table[s_t, a_t] = returns_sum[s_t, a_t] / returns_count[s_t, a_t]

    print(f"MC Ended. Avg Reward (last 100 eps): {np.mean(rewards_arr[-100:]):.2f}")
    # RETURN Q_TABLE TOO
    return rewards_arr, num_episodes_arr, q_table