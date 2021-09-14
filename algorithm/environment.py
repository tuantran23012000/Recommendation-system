import gym
from gym import spaces
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from action import ActionSpace
from state import StateSpace

class MyEnv(gym.Env):

    def __init__(self,
         history_data: pd.DataFrame,
         item_data: pd.DataFrame,
         user_data: pd.DataFrame,
         dim_action: int = 3,
         max_lag: int = 20,
  ): 
        
        super(MyEnv, self).__init__()
        self.history_data = history_data
        self.item_data = item_data
        self.user_data = user_data
        self.dim_action = dim_action
        self.max_lag = max_lag
        self.list_item = item_data.ID.tolist()
        self.n_item = len(self.list_item)
        self.encode = OneHotEncoder(handle_unknown='ignore')
        self.encode.fit(np.array(self.list_item).reshape(-1,1))
        self.action_space = ActionSpace(self.dim_action, self.n_item)
        self.observation_space = StateSpace(self.max_lag, self.n_item)
        self.idx_current = 0
        
    def step(self, action):
        action = np.array(action)
        _current_itemID = self.history_data.iloc[self.idx_current].ItemID
        _current_AcountID = self.history_data.iloc[self.idx_current].AccountID
        _temp = self.history_data.iloc[:self.idx_current + 1]
        current_frame = _temp[_temp.AccountID == _current_AcountID]
        if (len(current_frame) < self.max_lag):
            first_state = obs = np.zeros((self.max_lag - len(current_frame),self.n_item))
            str_obs = current_frame.ItemID.to_numpy().reshape(-1,1)
            last_state = self.encode.transform(str_obs).toarray()
            obs = np.concatenate([first_state, last_state],0)
        else:
            str_obs = current_frame[-self.max_lag:].ItemID.to_numpy().reshape(-1,1)
            obs = self.encode.transform(str_obs).toarray()
        
        _encode_current_itemID = self.encode.transform([[_current_itemID]]).toarray().reshape(-1)
        reward = 0
        for i in range(self.dim_action):
            if (action[i]==_encode_current_itemID).all():
                reward = self.dim_action - i
                break
        if (np.sum(action,1) > 1).any():
            reward = reward - 10
        done = False
        return obs, reward, done, {}
    def get_observation(self, reset = False):
        if reset:
            self.idx_current = np.random.randint(len(self.history_data))
        else:
            if (self.idx_current+1) == len(self.history_data):
                self.idx_current = 0
            else:
                self.idx_current = self.idx_current + 1
        _current_AcountID = self.history_data.iloc[self.idx_current].AccountID
        _temp = self.history_data.iloc[:self.idx_current]
        recent_past_frame = _temp[_temp.AccountID == _current_AcountID]
        
        first_state = obs = np.zeros((len(recent_past_frame),self.n_item))
        if (len(recent_past_frame) < self.max_lag):
            first_state = obs = np.zeros(( self.max_lag - len(recent_past_frame),self.n_item))
            str_obs = recent_past_frame.ItemID.to_numpy().reshape(-1,1)
            if len(str_obs) !=0:
                last_state = self.encode.transform(str_obs).toarray()
                obs = np.concatenate([first_state, last_state],0)
        else:
            str_obs = recent_past_frame[-self.max_lag:].ItemID.to_numpy().reshape(-1,1)
            obs = self.encode.transform(str_obs).toarray()
        return obs
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        raise Exception()