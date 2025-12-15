import sys
import os
from config import test_episodes
from utils import save_plots, plot_test_metrics

from q_learning import train_q_learning
from sarsa import train_sarsa
from monte_carlo import train_mc
from agent import evaluate_agent, save_model

def run_evaluation_and_plot(q_table, name):
    """Esegue il test e plotta subito i risultati"""
    rewards, steps, success = evaluate_agent(q_table, test_episodes, name=name)
    plot_test_metrics(rewards, steps, success, test_episodes, filename_prefix=name)

def main():
    print("-----------------------------------------")
    print("   TAXI-V3 RL MODULAR FRAMEWORK          ")
    print("-----------------------------------------")
    print("1. Run Q-Learning")
    print("2. Run SARSA")
    print("3. Run Monte Carlo (MC)")
    print("4. Run ALL and Compare")
    print("0. Exit")
    
    choice = input("\nSelect an option (0-4): ")

    if choice == '1':
        rewards, lengths, table = train_q_learning()
        save_plots(q_data=(rewards, lengths))
        run_evaluation_and_plot(table, "Q-Learning")
        
    elif choice == '2':
        rewards, lengths, table = train_sarsa()
        save_plots(sarsa_data=(rewards, lengths))
        run_evaluation_and_plot(table, "SARSA")

    elif choice == '3':
        rewards, lengths, table = train_mc()
        save_plots(mc_data=(rewards, lengths))
        run_evaluation_and_plot(table, "Monte_Carlo")

    elif choice == '4':
        print("\n--- Training ALL Algorithms ---")
        q_rew, q_len, q_tab = train_q_learning()
        run_evaluation_and_plot(q_tab, "Q-Learning")
        s_rew, s_len, s_tab = train_sarsa()
        run_evaluation_and_plot(s_tab, "SARSA")
        m_rew, m_len, m_tab = train_mc()
        run_evaluation_and_plot(m_tab, "Monte_Carlo")
        print("\nGenerating Comparison Plots...")
        save_plots(
            q_data=(q_rew, q_len), 
            sarsa_data=(s_rew, s_len), 
            mc_data=(m_rew, m_len)
        )

    elif choice == '0':
        sys.exit()
    
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()