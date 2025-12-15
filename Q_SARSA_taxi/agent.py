import numpy as np
from config import env, rng, max_episode_length

def evaluate_agent(q_table, episodes, name="Agent"):
    """
    Valuta l'agente addestrato per N episodi.
    Usa epsilon=0.1 per simulare il 'rumore' del worker HRL (confronto equo).
    """
    print(f"\n--- Testing {name} for {episodes} episodes (with Epsilon=0.1) ---")
    total_rewards = []
    total_steps = []
    success_count = 0
    test_epsilon = 0.1
    
    for _ in range(episodes):
        state, info = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        
        while not done:
            if rng.random() < test_epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            if steps > max_episode_length:
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        if terminated:
            success_count += 1
            
    print(f"Success Rate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
    return total_rewards, total_steps, success_count

def save_model(q_table, filename):
    """Salva la Q-Table su file .npy"""
    np.save(filename, q_table)
    print(f"Model saved to {filename}")

def load_model(filename):
    """Carica la Q-Table da file .npy"""
    print(f"Loading model from {filename}")
    return np.load(filename)