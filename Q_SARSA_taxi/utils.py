import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_training_metrics(rewards, steps, epsilons=None, filename_prefix="train", output_dir="plots"):
    """
    Plots comprehensive training metrics: Rewards, Steps, and Epsilon (optional).
    Adapted to use Seaborn and moving averages.
    """
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sns.set(style='darkgrid', font_scale=1.2)
    
    # 1. Learning Curve (Rewards)
    plt.figure(figsize=(12, 6))
    window = 100
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, color='green', linewidth=2, label=f'Moving Avg ({window})')
        # Also plot the raw data transparently in the background
        plt.plot(rewards, color='green', alpha=0.2, label='Raw Reward')
    else:
        plt.plot(rewards, color='green', alpha=0.6, label='Raw Reward')
        
    plt.title(f'{filename_prefix}: Average Reward per Episode')
    plt.ylabel('Reward')
    plt.xlabel('Episodes')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_learning_curve.png"))
    plt.close()

    # 2. Efficiency Curve (Steps per Episode)
    plt.figure(figsize=(12, 6))
    if len(steps) >= window:
        moving_avg_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg_steps, color='blue', linewidth=2, label=f'Moving Avg ({window})')
        plt.plot(steps, color='blue', alpha=0.2, label='Raw Steps')
    else:
        plt.plot(steps, color='blue', alpha=0.6, label='Raw Steps')
        
    plt.title(f'{filename_prefix}: Steps needed to Solve')
    plt.ylabel('Steps')
    plt.xlabel('Episodes')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_steps_curve.png"))
    plt.close()
    
    # 3. Epsilon Decay (Only if data provided)
    if epsilons is not None and len(epsilons) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(epsilons, color='orange', linewidth=2)
        plt.title(f'{filename_prefix}: Epsilon Decay over Time')
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon value')
        plt.savefig(os.path.join(output_dir, f"{filename_prefix}_epsilon_decay.png"))
        plt.close()
    
    print(f" -> Saved training plots for {filename_prefix}")

def plot_test_metrics(rewards, steps, success_count, total_episodes, filename_prefix="test", output_dir="plots"):
    """
    Plots statistical test metrics: Distributions and Success Rate.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sns.set(style='whitegrid', font_scale=1.2)
    
    # 1. Reward Distribution (Histogram)
    plt.figure(figsize=(10, 6))
    sns.histplot(rewards, kde=True, color='purple', bins=20)
    plt.title(f'{filename_prefix}: Reward Distribution (Consistency Check)')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_reward_dist.png"))
    plt.close()

    # 2. Steps Distribution (Efficiency Check)
    plt.figure(figsize=(10, 6))
    sns.histplot(steps, kde=True, color='teal', bins=20)
    plt.title(f'{filename_prefix}: Steps Distribution')
    plt.xlabel('Steps to Solve')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_steps_dist.png"))
    plt.close()
    
    # 3. Success Rate (Pie Chart)
    labels = ['Success', 'Failure']
    sizes = [success_count, total_episodes - success_count]
    colors = ['#66b3ff', '#ff9999']
    
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'{filename_prefix}: Success Rate ({total_episodes} Episodes)')
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_success_pie.png"))
    plt.close()
    
    print(f" -> Saved test plots for {filename_prefix}")

def save_plots(q_data=None, sarsa_data=None, mc_data=None):
    """
    Main function to save plots using the new Seaborn style.
    Handles individual algorithm plots and a final comparison.
    """
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")

    print(f"Saving plots to '{output_dir}' using Seaborn styles...")

    # --- 1. Individual Plots using plot_training_metrics ---
    
    if q_data is not None:
        rewards, lengths = q_data
        # Note: We pass None for epsilons as we don't track them yet in train.py
        plot_training_metrics(rewards, lengths, epsilons=None, filename_prefix="Q-Learning", output_dir=output_dir)

    if sarsa_data is not None:
        rewards, lengths = sarsa_data
        plot_training_metrics(rewards, lengths, epsilons=None, filename_prefix="SARSA", output_dir=output_dir)

    if mc_data is not None:
        rewards, lengths = mc_data
        plot_training_metrics(rewards, lengths, epsilons=None, filename_prefix="Monte_Carlo", output_dir=output_dir)

    # --- 2. Comparison Plots (If all data is present) ---
    if q_data is not None and sarsa_data is not None and mc_data is not None:
        sns.set(style='darkgrid', font_scale=1.2)
        
        # Truncate to min length
        min_len = min(len(q_data[0]), len(sarsa_data[0]), len(mc_data[0]))
        window = 100
        
        # Helper to compute moving average safely
        def get_moving_avg(data, w):
            if len(data) < w: return data
            return np.convolve(data, np.ones(w)/w, mode='valid')

        # COMPARISON: REWARDS
        plt.figure(figsize=(12, 6))
        
        q_avg = get_moving_avg(q_data[0][:min_len], window)
        s_avg = get_moving_avg(sarsa_data[0][:min_len], window)
        m_avg = get_moving_avg(mc_data[0][:min_len], window)
        
        plt.plot(q_avg, label="Q-Learning", linewidth=2)
        plt.plot(s_avg, label="SARSA", linewidth=2)
        plt.plot(m_avg, label="Monte Carlo", linewidth=2)
        
        plt.title(f'Comparison: Average Reward (Moving Avg {window})')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "comparison_rewards.png"))
        plt.close()

        # COMPARISON: STEPS
        plt.figure(figsize=(12, 6))
        
        q_steps_avg = get_moving_avg(q_data[1][:min_len], window)
        s_steps_avg = get_moving_avg(sarsa_data[1][:min_len], window)
        m_steps_avg = get_moving_avg(mc_data[1][:min_len], window)
        
        plt.plot(q_steps_avg, label="Q-Learning", linewidth=2)
        plt.plot(s_steps_avg, label="SARSA", linewidth=2)
        plt.plot(m_steps_avg, label="Monte Carlo", linewidth=2)
        
        plt.title(f'Comparison: Steps per Episode (Moving Avg {window})')
        plt.xlabel('Episodes')
        plt.ylabel('Steps')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "comparison_steps.png"))
        plt.close()
        
        print(" -> Saved comparison plots.")