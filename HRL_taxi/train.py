# train.py
import gymnasium as gym
import numpy as np
import config
from agent import RobustDiscoveryAgent
from utils import plot_training_metrics # Import the updated plotting function

def run_episode_logic(env, agent, i, visualize=False, seed=None):
    """
    Executes an episode keeping the HRL logic intact.
    Returns: reward_sum, total_steps, epsilon
    """
    state, _ = env.reset(seed=seed) 
    agent.done = False
    r_sum = 0
    total_steps = 0
    
    # Epsilon calculation
    epsilon = max(config.EPSILON_MIN, 
                  config.EPSILON_START - (i / config.EPSILON_DECAY_STEPS))
    
    while not agent.done:
        option_idx = agent.get_meta_action(state, epsilon=epsilon)
        
        # Execute option (returns steps for that specific option)
        # We pass visualize to activate time.sleep if necessary
        next_state, reward, steps = agent.execute_option(
            option_idx, state, is_training=True, visualize=visualize
        )
        r_sum += reward
        total_steps += steps # Accumulate total steps for the episode
        
        # SMDP Update
        best_next = np.argmax(agent.Q_meta[next_state, :])
        target = reward + (agent.gamma ** steps) * agent.Q_meta[next_state, best_next]
        agent.Q_meta[state, option_idx] += agent.alpha * (target - agent.Q_meta[state, option_idx])
        
        state = next_state
        if r_sum < -1000: break
    
    return r_sum, total_steps, epsilon

def main():
    # Fast Environment (Default)
    env_fast = gym.make(config.ENV_NAME)
    agent = RobustDiscoveryAgent(env_fast)
    
    # Data collection lists for plotting
    rewards_history = []
    steps_history = []
    epsilon_history = []
    
    print(f" Starting Hybrid Training on {config.TOTAL_EPISODES} episodes...")
    print(f" Visual Checkpoints: {config.VISUAL_CHECKPOINTS}")

    for i in range(config.TOTAL_EPISODES):
        
        # --- IS THIS AN EPISODE TO SHOW? ---
        if i in config.VISUAL_CHECKPOINTS:
            print(f"\n [Episode {i}] Opening graphics window...")
            
            # 1. Create temporary graphical environment
            env_vis = gym.make(config.ENV_NAME, render_mode='human')
            agent.env = env_vis # Swap the environment in the agent
            
            # 2. Run the SLOW episode with FIXED SEED
            # Note: We capture 3 return values now (r, s, eps)
            r, s, eps = run_episode_logic(env_vis, agent, i, visualize=True, seed=config.DEMO_SEED)
            
            # 3. Close and go back to fast
            env_vis.close()
            agent.env = env_fast 
            
            print(f" Episode {i} completed. Reward: {r}, Steps: {s}")
            
        else:
            # --- NORMAL EPISODE (FAST) ---
            r, s, eps = run_episode_logic(env_fast, agent, i, visualize=False, seed=None)
            
        # Store metrics for graphs
        rewards_history.append(r)
        steps_history.append(s)
        epsilon_history.append(eps)
        
        # Logging every 100 episodes
        if i % 100 == 0:
            avg = np.mean(rewards_history[-100:]) if i >= 100 else np.mean(rewards_history)
            print(f"Ep {i} | Avg Reward: {avg:.2f} | Steps: {s} | Eps: {eps:.2f}")

    agent.save(config.MODEL_FILENAME)
    print(" Model saved.")
    
    # Generate all training graphs
    print(" Generating Training Graphs...")
    plot_training_metrics(rewards_history, steps_history, epsilon_history)

if __name__ == "__main__":
    main()