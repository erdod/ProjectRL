ENV_NAME = 'Taxi-v3'
LOCS = [(0,0), (0,4), (4,0), (4,3)] 

OPTIONS = {
    0: "Go to Red", 1: "Go to Green", 2: "Go to Blue", 3: "Go to Yellow",
    4: "Pickup", 5: "Dropoff" 
}
NUM_OPTIONS = len(OPTIONS)

ALPHA = 0.4       
GAMMA = 0.99      
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY_STEPS = 3000 

TOTAL_EPISODES = 5001 
MODEL_FILENAME = "taxi_hrl_brain.pkl"

VISUAL_CHECKPOINTS = [0, 100, 500, 2000, 3000, 5000]

DEMO_SEED = 42