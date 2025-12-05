# config.py

# --- ENVIRONMENT PARAMETERS ---
ENV_NAME = 'Taxi-v3'
LOCS = [(0,0), (0,4), (4,0), (4,3)] 

OPTIONS = {
    0: "Go to Red", 1: "Go to Green", 2: "Go to Blue", 3: "Go to Yellow",
    4: "Pickup", 5: "Dropoff" 
}
NUM_OPTIONS = len(OPTIONS)

# --- HRL HYPERPARAMETERS (LOGIC UNCHANGED) ---
ALPHA = 0.4       
GAMMA = 0.99      
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY_STEPS = 3000 

# --- VISUALIZATION SETTINGS ---
# Set to 5001 to ensure episode 5000 is executed
TOTAL_EPISODES = 5001 
MODEL_FILENAME = "taxi_hrl_brain.pkl"

# EPISODES WHERE YOU WANT TO SEE THE TAXI
VISUAL_CHECKPOINTS = [0, 100, 500, 2000, 3000, 5000]

# We use a fixed seed ONLY during visualization to see 
# how it improves on the SAME problem.
DEMO_SEED = 42