import numpy as np
import gymnasium as gym

from collections import namedtuple
ImgDim = namedtuple('ImgDim', 'width height')

# Base environment implementing basic discounting and reward calcuation
class BBO(gym.Env):
    def __init__(self, naive, step_size, max_num_step, seed=42):
        # If naive = True, use a simple reward = -val. If naive = False, use a more advanced reward shaping method PMP-based shaping
        self.naive = naive
        
        # Discount info
        # Gamma is the discount factor (how much future steps are worth)
        self.gamma = 0.99
        # step_pow is a power (default 1.0). Lets  you customize the discount rate
        self.step_pow = 1.0
        # gamma_inc = gamma ^ step_pow.
        self.gamma_inc = self.gamma**self.step_pow
        # discount starts at 1.0 and gets multiplied over time. 
        self.discount = 1.0

        # Step info
        # max_num_step: the episode ends after this many steps
        self.max_num_step = max_num_step
        # num_step: keeps track of how many steps have been taken
        self.num_step = 0
        # step_size: how strongly actions affect the shape
        self.step_size = step_size

        # Creates a random number generation with a given seed. Ensures reproducibility, running with the same seed gives you same random shapes
        self.rng = np.random.default_rng(seed=seed)

    def set_gamma(self, gamma):
        # Update the discount factor (gamma) to a new value
        self.gamma = gamma
        # Reset the power used for discount accumulation
        self.step_pow = 1.0
        # Precompute gamma^step_pow for reuse later
        self.gamma_inc = self.gamma**self.step_pow
        # Reset discount accumulation to 1.0
        self.discount = 1.0

    def calculate_final_reward(self, val, action):
        # Convert raw geometric value and aciton magnitude into scalar reward
        if self.naive:
            # Simple: negative of the objective
            reward = -val
        else:
            # PMP-based reward: penalize large actions, discount over time
            self.discount *= self.gamma_inc
            # reward = (0.5 * ||action||^2 - val) / (discount^2)
            reward = 1/(self.discount**2) * (np.sum(action**2)*0.5 - val)

        return reward
    
    def get_val(self, reward, action):
        """
        Reverses the reward calculation to figure out what the original val must have been
        """
        if self.naive:
            # Simple: negative of the objective
            return -reward 
        else:
            # Inverts shaped reward formula to solve for val
            return np.sum(action**2)*0.5 - (reward * (self.discount**2))