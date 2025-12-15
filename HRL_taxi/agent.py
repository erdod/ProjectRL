import numpy as np
import pickle
import time
import config 

class RobustDiscoveryAgent:
    def __init__(self, env):
        self.env = env
        self.alpha = config.ALPHA
        self.gamma = config.GAMMA
        
        self.locs = config.LOCS
        self.options = config.OPTIONS
        self.num_options = config.NUM_OPTIONS
        
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        
        self.Q_meta = np.zeros((self.nS, self.num_options))
        self.Q_options = np.zeros((self.num_options, self.nS, self.nA))
        
        self.done = False

    def check_option_termination(self, state, option_idx):
        taxirow, taxicol, passidx, destidx = list(self.env.unwrapped.decode(state))
        
        if option_idx <= 3: 
            target_loc = self.locs[option_idx]
            if taxirow == target_loc[0] and taxicol == target_loc[1]: return True, 10.0 
        elif option_idx == 4: 
            if passidx == 4: return True, 50.0 
        elif option_idx == 5: 
            if passidx == destidx and (taxirow, taxicol) == self.locs[destidx]: return True, 100.0 
        
        if self.done: return True, -1.0
        return False, 0.0

    def get_primitive_action(self, state, option_idx, epsilon=0.1):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.Q_options[option_idx, state, :])

    def get_meta_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.choice(self.num_options)
        return np.argmax(self.Q_meta[state, :])

    def execute_option(self, option_idx, state, is_training=True, visualize=False):
        steps = 0
        cumulative_reward = 0
        option_terminated = False
        epsilon_worker = 0.1 if is_training else 0.0
        
        while not option_terminated and not self.done:
            
            if visualize: time.sleep(0.05) 
            
            action = self.get_primitive_action(state, option_idx, epsilon_worker)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.done = terminated or truncated
            
            option_terminated, internal_reward = self.check_option_termination(next_state, option_idx)
            
            if is_training:
                r_worker = internal_reward - 0.1 
                best_next_a = np.argmax(self.Q_options[option_idx, next_state, :])
                td_target = r_worker + self.gamma * self.Q_options[option_idx, next_state, best_next_a]
                self.Q_options[option_idx, state, action] += self.alpha * (td_target - self.Q_options[option_idx, state, action])
            
            cumulative_reward += reward
            state = next_state
            steps += 1
            if steps > 500: break 
                
        return state, cumulative_reward, steps

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump({"Q_meta": self.Q_meta, "Q_options": self.Q_options}, f)
            
    def load(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.Q_meta = data["Q_meta"]
            self.Q_options = data["Q_options"]