import numpy as np
from config import env, max_episode_length, rng, params_q

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
    return rewards_arr, num_episodes_arr, q_table