from train import train_q_learning, train_sarsa, train_mc
from utils import plot_results
from config import N

def main():
    """
    Main function to run the reinforcement learning algorithms and plot the results.
    """
    q_rewards, q_lengths = train_q_learning()
    sarsa_rewards, sarsa_lengths = train_sarsa()
    
    # Note: The original MC algorithm runs for 1000 episodes, not N (10000)
    # The plotting function will adjust, but the arrays will be of different lengths.
    mc_rewards, mc_lengths = train_mc()
    
    # We need to decide how to plot episodes of different lengths.
    # For now, we'll use N, but the MC arrays will be shorter.
    # A better approach might be to use the length of the shortest array.
    # Let's use the min length for plotting to avoid errors.
    plot_n = min(len(q_rewards), len(sarsa_rewards), len(mc_rewards))

    plot_results(q_rewards, q_lengths, sarsa_rewards, sarsa_lengths, mc_rewards, mc_lengths, plot_n)

if __name__ == "__main__":
    main()
