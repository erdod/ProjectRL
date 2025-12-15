import numpy as np
from config import env, max_episode_length, rng, params_sarsa

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
    return rewards_arr, num_episodes_arr, q_table