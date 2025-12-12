import sys
from train import train_q_learning, train_sarsa, train_mc, evaluate_agent
from utils import save_plots, plot_test_metrics
from config import test_episodes

def run_evaluation_and_plot(q_table, name):
    """
    Helper function to run test episodes and plot test metrics immediately.
    """
    rewards, steps, success = evaluate_agent(q_table, test_episodes, name=name)
    plot_test_metrics(rewards, steps, success, test_episodes, filename_prefix=name)

def main():
    print("-----------------------------------------")
    print("   TAXI-V3 REINFORCEMENT LEARNING MENU   ")
    print("-----------------------------------------")
    print("1. Run Q-Learning")
    print("2. Run SARSA")
    print("3. Run Monte Carlo (MC)")
    print("4. Run ALL and Compare")
    print("0. Exit")
    
    choice = input("\nSelect an option (0-4): ")

    if choice == '1':
        # Q-Learning
        q_rewards, q_lengths, q_table = train_q_learning()
        save_plots(q_data=(q_rewards, q_lengths))
        run_evaluation_and_plot(q_table, "Q-Learning")
        
    elif choice == '2':
        # SARSA
        s_rewards, s_lengths, s_table = train_sarsa()
        save_plots(sarsa_data=(s_rewards, s_lengths))
        run_evaluation_and_plot(s_table, "SARSA")

    elif choice == '3':
        # MC
        m_rewards, m_lengths, m_table = train_mc()
        save_plots(mc_data=(m_rewards, m_lengths))
        run_evaluation_and_plot(m_table, "Monte_Carlo")

    elif choice == '4':
        # Run All
        print("\nStarting full execution...")
        
        # Train and Test Q-Learning
        q_rewards, q_lengths, q_table = train_q_learning()
        run_evaluation_and_plot(q_table, "Q-Learning")
        
        # Train and Test SARSA
        s_rewards, s_lengths, s_table = train_sarsa()
        run_evaluation_and_plot(s_table, "SARSA")
        
        # Train and Test MC
        m_rewards, m_lengths, m_table = train_mc()
        run_evaluation_and_plot(m_table, "Monte_Carlo")
        
        # Save Comparison Training Plots
        save_plots(
            q_data=(q_rewards, q_lengths), 
            sarsa_data=(s_rewards, s_lengths), 
            mc_data=(m_rewards, m_lengths)
        )

    elif choice == '0':
        print("Exiting.")
        sys.exit()
    
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()