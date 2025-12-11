import sys
from train import train_q_learning, train_sarsa, train_mc
from utils import save_plots

def main():
    print("-----------------------------------------")
    print("   TAXI-V3 REINFORCEMENT LEARNING MENU   ")
    print("-----------------------------------------")
    print("1. Esegui Q-Learning")
    print("2. Esegui SARSA")
    print("3. Esegui Monte Carlo (MC)")
    print("4. Esegui TUTTI e crea confronto")
    print("0. Esci")
    
    choice = input("\nSeleziona un'opzione (0-4): ")

    if choice == '1':
        # Solo Q-Learning
        q_rewards, q_lengths = train_q_learning()
        save_plots(q_data=(q_rewards, q_lengths))
        
    elif choice == '2':
        # Solo SARSA
        s_rewards, s_lengths = train_sarsa()
        save_plots(sarsa_data=(s_rewards, s_lengths))

    elif choice == '3':
        # Solo MC
        m_rewards, m_lengths = train_mc()
        save_plots(mc_data=(m_rewards, m_lengths))

    elif choice == '4':
        # Tutti
        print("\nAvvio esecuzione completa...")
        q_rewards, q_lengths = train_q_learning()
        s_rewards, s_lengths = train_sarsa()
        m_rewards, m_lengths = train_mc()
        
        # Passiamo tutto a save_plots
        save_plots(
            q_data=(q_rewards, q_lengths), 
            sarsa_data=(s_rewards, s_lengths), 
            mc_data=(m_rewards, m_lengths)
        )

    elif choice == '0':
        print("Uscita.")
        sys.exit()
    
    else:
        print("Scelta non valida.")

if __name__ == "__main__":
    main()