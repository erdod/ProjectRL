import time
import numpy as np
import config
import gymnasium as gym
from utils import plot_test_metrics

def execution_view(agent):
    """
    Executes a single visual episode (formerly demo_visiva_ragionata).
    """
    env = gym.make(config.ENV_NAME, render_mode='human')
    agent.env = env
    
    print("\n" + "="*60)
    print("STARTING EXECUTION VIEW (Graph Analysis)")
    print("="*60)
    
    state, _ = env.reset(seed=42) 
    agent.done = False
    agent.r_sum = 0
    macro_step = 0
    
    r, c, p, d = env.unwrapped.decode(state)
    p_str = ["R", "G", "B", "Y", "Taxi"][p]
    d_str = ["R", "G", "B", "Y"][d]
    print(f"START: Taxi({r},{c}) | Pass: {p_str} | Dest: {d_str}")
    
    while not agent.done:
        option_idx = agent.get_meta_action(state, epsilon=0.0)
        opt_name = config.OPTIONS[option_idx]
        
        next_state, reward, steps = agent.execute_option(option_idx, state, is_training=False, visualize=True)
        
        agent.r_sum += reward
        macro_step += 1
        
        r_new, c_new, p_new, d_new = env.unwrapped.decode(next_state)
        p_new_str = ["R", "G", "B", "Y", "Taxi"][p_new]
        
        print(f"\n MACRO STEP {macro_step}: [{opt_name}]")
        print(f"   Executed in {steps} steps -> Taxi({r_new},{c_new}), Pass: {p_new_str}")
        
        if p_new != p:
            print(f"PASSENGER STATE CHANGED!")
            p = p_new
        
        state = next_state
        if macro_step > 15: break
        
    print("-" * 60)
    if agent.r_sum > 0:
        print(f" SUCCESS! Final Reward: {agent.r_sum}")
    else:
        print(f" FAILURE. Final Reward: {agent.r_sum}")
    
    env.close()


def test(agent, episodes=100):
    """
    Executes N episodes quickly to calculate performance metrics
    and generate test graphs (formerly valutazione_statistica).
    """
    env = gym.make(config.ENV_NAME) 
    agent.env = env
    
    print(f"\n STARTING STATISTICAL TEST ON {episodes} EPISODES...")
    
    success_count = 0
    rewards_list = []
    steps_list = []
    
    for i in range(episodes):
        state, _ = env.reset()
        agent.done = False
        total_reward = 0
        total_steps = 0
        
        while not agent.done:
            option_idx = agent.get_meta_action(state, epsilon=0.0)
            
            next_state, reward, steps = agent.execute_option(option_idx, state, is_training=False, visualize=False)
            
            total_reward += reward
            total_steps += steps
            state = next_state
            
            if total_reward < -100: break
            
        rewards_list.append(total_reward)
        steps_list.append(total_steps)
        
        if total_reward > 0:
            success_count += 1
            
    avg_rew = np.mean(rewards_list)
    avg_steps = np.mean(steps_list)
    success_rate = (success_count / episodes) * 100
    
    print("-" * 40)
    print(f" RESULTS:")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Avg Reward:   {avg_rew:.2f}")
    print(f"   Avg Steps:    {avg_steps:.2f}")
    print("-" * 40)
    print(" Generating Test Graphs...")
    plot_test_metrics(rewards_list, steps_list, success_count, episodes)