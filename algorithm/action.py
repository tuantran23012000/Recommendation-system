import torch
import numpy as np
import gym

class ActionSpace(gym.Space):
    def __init__(self, n_reco, n_item):
        self.shape = (n_reco, n_item)
        self.dtype = np.int64
        self.low = 0
        self.high = 1
        super(ActionSpace, self).__init__(self.shape,self.dtype)
    def sample(self):
        sample = torch.zeros(self.shape,torch.int64)
        indices = torch.randint(0,n_item,(n_reco,1))
        sampe = sample.scatter_(1,indices,1)
        return sampe.numpy()