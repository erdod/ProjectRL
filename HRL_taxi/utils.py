import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_metrics(rewards, steps, epsilons, filename_prefix="train"):
    """
    Plots comprehensive training metrics: Rewards, Steps, and Epsilon.
    """
    sns.set(style='darkgrid', font_scale=1.2)
    plt.figure(figsize=(12, 6))
    window = 100
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, color='green', linewidth=2, label='Moving Avg (100)')
        plt.xlabel(f'Episodes')
    else:
        plt.plot(rewards, color='green', alpha=0.6)
        plt.xlabel('Episodes')
    
    plt.title('Training: Average Reward per Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(f"{filename_prefix}_learning_curve.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    if len(steps) >= window:
        moving_avg_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg_steps, color='blue', linewidth=2, label='Moving Avg (100)')
    else:
        plt.plot(steps, color='blue', alpha=0.6)
        
    plt.title('Training: Steps needed to Solve')
    plt.ylabel('Steps')
    plt.xlabel('Episodes')
    plt.legend()
    plt.savefig(f"{filename_prefix}_steps_curve.png")
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(epsilons, color='orange', linewidth=2)
    plt.title('Training: Epsilon Decay over Time')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon value')
    plt.savefig(f"{filename_prefix}_epsilon_decay.png")
    plt.close()
    
    print(f"✅ Training plots saved with prefix '{filename_prefix}_'")

def plot_test_metrics(rewards, steps, success_count, total_episodes, filename_prefix="test"):
    """
    Plots statistical test metrics: Distributions and Success Rate.
    """
    sns.set(style='whitegrid', font_scale=1.2)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(rewards, kde=True, color='purple', bins=20)
    plt.title('Test: Reward Distribution (Consistency Check)')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.savefig(f"{filename_prefix}_reward_dist.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(steps, kde=True, color='teal', bins=20)
    plt.title('Test: Steps Distribution')
    plt.xlabel('Steps to Solve')
    plt.ylabel('Frequency')
    plt.savefig(f"{filename_prefix}_steps_dist.png")
    plt.close()

    labels = ['Success', 'Failure']
    sizes = [success_count, total_episodes - success_count]
    colors = ['#66b3ff', '#ff9999']
    
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'Test: Success Rate ({total_episodes} Episodes)')
    plt.savefig(f"{filename_prefix}_success_pie.png")
    plt.close()
    
    print(f" Test plots saved with prefix '{filename_prefix}_'")