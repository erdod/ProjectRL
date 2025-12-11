import numpy as np
from config import env, alpha, gamma, epsilon, max_episode_length, N, rng
from agent import choose_action

def train_q_learning():
    """
    Trains the agent using the Q-Learning algorithm.
    """
    print("Q-Learning Started")
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards_arr = []
    num_episodes_arr = []
    
    lst = list(range(1, N + 1))
    for i in lst:
        rewards = []
        state, info = env.reset()
        done = False
        lgt = 0
        while not done:
            if lgt > max_episode_length:
                break
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, done, flag, info = env.step(action)
            current_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            rewards.append(reward)
            q_table[state, action] = (1 - alpha) * current_value + alpha * (reward + gamma * next_max)
            state = next_state
            lgt += 1
        
        rewards_arr.append(sum(rewards) / len(rewards))
        num_episodes_arr.append(lgt)

    print("Q-Table for Q-Learning: ")
    print(q_table)
    print("Average Reward for Q-Learning Algorithm: ", np.mean(rewards_arr))
    print("Q-Learning Ended")
    return rewards_arr, num_episodes_arr

def train_sarsa():
    """
    Trains the agent using the SARSA algorithm.
    """
    print("SARSA Started")
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards_arr = []
    num_episodes_arr = []
    
    lst = list(range(1, N + 1))
    for i in lst:
        state, info = env.reset()
        rewards = []
        done = False
        lgt = 0
        action = choose_action(state, q_table)
        while not done:
            next_state, reward, done, flag, info = env.step(action)
            current_value = q_table[state, action]
            next_action = choose_action(next_state, q_table)
            next_val = q_table[next_state, next_action]
            rewards.append(reward)
            q_table[state, action] = (1 - alpha) * current_value + alpha * (reward + gamma * next_val)
            state = next_state
            action = next_action
            if lgt > max_episode_length:
                break
            lgt += 1
        
        rewards_arr.append(sum(rewards) / len(rewards))
        num_episodes_arr.append(lgt)

    print("Q-Table for SARSA: ")
    print(q_table)
    print("Average Reward for SARSA: ", np.mean(rewards_arr))
    print("SARSA ended")
    return rewards_arr, num_episodes_arr

def train_mc():
    """
    Trains the agent using the Monte Carlo Every-Visit algorithm.
    """
    print("MC Started")
    v_table = np.zeros([env.observation_space.n])
    policy = dict()
    returns = dict()
    for i in range(0, 500):
        policy[i] = env.action_space.sample(env.action_mask(i))
        returns[i] = []

    rewards_arr = []
    num_episodes_arr = []
    
    # In the original code, N for MC is 1000, not 10000
    mc_n = 1000
    lst = list(range(1, mc_n + 1))
    
    for k in lst:
        lgt = 0
        done = False
        state, info = env.reset()
        episode = [state]
        episodeReward = [0]
        rnd = False
        while not done:
            if k == 1 or rnd:
                action = env.action_space.sample(env.action_mask(state))
            else:
                action = policy[state]
            
            next_state, reward, done, flag, info = env.step(action)
            if done:
                break
            episode.append(next_state)
            episodeReward.append(reward)
            state = next_state
            if state == next_state:
                rnd = True
            else:
                rnd = False
            if lgt > max_episode_length:
                break
            lgt += 1
        
        num_episodes_arr.append(lgt)

        G = 0
        for i in range(len(episode)):
            G = episodeReward[i] + gamma * G
            returns[episode[i]].append(G)
            tmp = v_table[episode[i]]
            v_table[episode[i]] = sum(returns[episode[i]]) / len(returns[episode[i]])

        for z in range(0, 500):
            a = policy[z]
            max_value = -float('inf')
            max_action = None
            for action in range(6):
                # The original logic for policy improvement seems to have a bug where it doesn't use the environment correctly.
                # Replicating original behavior.
                # A proper implementation would likely involve creating a copy of the env or resetting to the state `z`.
                # For now, this just uses the last `next_state` from the episode generation.
                next_state_pi, reward_pi, _, _, _ = env.step(action)
                tmp = reward_pi
                if tmp > max_value:
                    max_value = tmp
                    max_action = action
            policy[z] = max_action
        
        # Avoid division by zero if episodeReward is empty
        if len(episodeReward) > 0:
            rewards_arr.append(sum(episodeReward) / len(episodeReward))
        else:
            rewards_arr.append(0)


    print("V(s): ")
    print(v_table)
    print("Average Reward for MC Every-Visit: ", np.mean(rewards_arr))
    print("MC Ended")
    return rewards_arr, num_episodes_arr
