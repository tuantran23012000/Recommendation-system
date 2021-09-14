import torch
import numpy as np

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.1, max_sigma=0.5, min_sigma=0.0, decay_period=500):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim[0],self.action_dim[1])
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0): 
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        action = np.clip(action + ou_state, self.low, self.high)
        action = torch.from_numpy(action)
        action = torch.nn.Softmax(dim=-1)(action).detach().numpy()
        return action