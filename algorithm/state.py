import torch
import numpy as np
import gym

class StateSpace(gym.Space):
    def __init__(self, max_state, n_item):
        self.shape = (max_state, n_item)
        self.dtype = np.int64
        super(StateSpace, self).__init__(self.shape,self.dtype)