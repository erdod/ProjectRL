from train import train_q_learning, train_sarsa, train_mc
from utils import plot_results
# RIMOSSO: from config import N  <-- Questo causava l'errore perché N non esiste più lì.

def main():
    """
    Main function to run the reinforcement learning algorithms and plot the results.
    """
    # 1. Eseguiamo i training.
    # I parametri (episodi, alpha, ecc.) sono ora presi automaticamente da config.py dentro queste funzioni.
    q_rewards, q_lengths = train_q_learning()
    sarsa_rewards, sarsa_lengths = train_sarsa()
    mc_rewards, mc_lengths = train_mc()
    
    # 2. Gestione del Plotting
    # Dato che ora gli algoritmi hanno lunghezze diverse (es. Q=5000, SARSA=10000, MC=20000),
    # dobbiamo decidere come passarli alla funzione di plot.
    
    # Calcoliamo la lunghezza minima per evitare errori di indice se la funzione plot non gestisce lunghezze variabili
    min_length = min(len(q_rewards), len(sarsa_rewards), len(mc_rewards))
    
    print(f"\n--- Risultati pronti per il grafico ---")
    print(f"Episodi Q-Learning: {len(q_rewards)}")
    print(f"Episodi SARSA: {len(sarsa_rewards)}")
    print(f"Episodi MC: {len(mc_rewards)}")
    print(f"Plotting troncato ai primi {min_length} episodi per confronto diretto...")

    # Passiamo i dati alla funzione di plot.
    # Nota: Stiamo passando 'min_length' come parametro N. 
    # Assicurati che plot_results in utils.py usi questo numero per tagliare le array o settare l'asse X.
    plot_results(q_rewards, q_lengths, sarsa_rewards, sarsa_lengths, mc_rewards, mc_lengths, min_length)

if __name__ == "__main__":
    main()