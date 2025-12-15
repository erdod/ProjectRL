import numpy as np
from config import env, max_episode_length, rng, params_mc

def train_mc():
    print("\n--- Monte Carlo Started ---")
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
    return rewards_arr, num_episodes_arr, q_table