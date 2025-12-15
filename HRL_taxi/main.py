# main.py
import gymnasium as gym
import config
from agent import RobustDiscoveryAgent
from test_functions import execution_view, test

def main():
    env_placeholder = gym.make(config.ENV_NAME)
    agent = RobustDiscoveryAgent(env_placeholder)
    
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
            test(agent, episodes=100)
            
        elif choice == "2":
            execution_view(agent)
            
        elif choice == "3":
            print("Closing.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()