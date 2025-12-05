# main.py
import gymnasium as gym
import config
from agent import RobustDiscoveryAgent
from test_functions import execution_view, test

def main():
    # 1. Agent Initialization (with placeholder environment)
    # The actual environment is set inside the test functions
    env_placeholder = gym.make(config.ENV_NAME)
    agent = RobustDiscoveryAgent(env_placeholder)
    
    # 2. Loading the Brain
    try:
        print(f" Loading model from '{config.MODEL_FILENAME}'...")
        agent.load(config.MODEL_FILENAME)
        print(" Model loaded successfully.")
    except FileNotFoundError:
        print(" ERROR: Model file not found.")
        print("   Run 'python train.py' first to train the agent.")
        return

    while True:
        print("\n" + "="*30)
        print("   TAXI HRL TEST MENU")
        print("="*30)
        print("1.  Test (100 ep)")
        print("2.  Execution-view (Render Human)")
        print("3.  Exit")
        
        choice = input("\nChoose an option (1-3): ")
        
        if choice == "1":
            # Pass the agent to the statistical function
            test(agent, episodes=100)
            
        elif choice == "2":
            # Pass the agent to the visual function
            execution_view(agent)
            
        elif choice == "3":
            print("Closing.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()