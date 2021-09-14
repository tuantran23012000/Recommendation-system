import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from src.model import *
from utils import *
from memory_buffer import Memory
import os

PARENT_PATH = 'weight'
ACTOR_PATH = 'weight/actor'
ACTOR_TARGET_PATH = 'weight/actor_target'
CRITIC_PATH = 'weight/critic'
CRITIC_TARGET_PATH = 'weight/critic_target'

class DDPGagent:
    def __init__(self, env, hidden_size=576, 
                 actor_learning_rate=1e-4, 
                 critic_learning_rate=1e-3, 
                 gamma=0.99, tau=1e-2, 
                 max_memory_size=50000):
        # Params
        self.size_states = env.observation_space.shape
        self.size_actions = env.action_space.shape
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = Actor(self.size_states[1],self.size_actions[0], hidden_size, self.size_actions[1])
        self.actor_target = Actor(self.size_states[1],self.size_actions[0], hidden_size, self.size_actions[1])
        self.critic = Critic(self.size_states[1] ,self.size_actions[1] , hidden_size, self.size_actions[0])
        self.critic_target = Critic(self.size_states[1] ,self.size_actions[1] , hidden_size, self.size_actions[0])

        self.load_()
        
        # Training
        self.memory = Memory(max_memory_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False
    
    def from_probability_distribution_to_action(self,action):
        if not isinstance(action,torch.Tensor):
            action = torch.FloatTensor(action)
        indices = torch.max(action,-1).indices.unsqueeze(-1)
        action = action.zero_().scatter_(-1,indices,1).numpy()
        return action
    
    def get_action(self, state):
        if not isinstance(state,torch.Tensor):
            state = torch.FloatTensor(state)
        with torch.no_grad():
            action = self.actor.forward(state)
        action = action.detach().numpy()
        return action
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
    
        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_actions = self.from_probability_distribution_to_action(next_actions)
        next_Q = self.critic_target.forward(next_states, next_actions)
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks
        with torch.no_grad():
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
    def save_(self):
        if not os.path.exists(PARENT_PATH):
            os.mkdir(PARENT_PATH)
        torch.save(self.actor.state_dict(), ACTOR_PATH)
        torch.save(self.actor_target.state_dict(), ACTOR_TARGET_PATH)
        torch.save(self.critic.state_dict(), CRITIC_PATH)
        torch.save(self.critic_target.state_dict(), CRITIC_TARGET_PATH)
    def load_(self):
        try:
            self.actor.load_state_dict(torch.load(ACTOR_PATH))
            self.actor_target.load_state_dict(torch.load(ACTOR_TARGET_PATH))
            self.critic.load_state_dict(torch.load(CRITIC_PATH))
            self.critic_target.load_state_dict(torch.load(CRITIC_TARGET_PATH))
        except Exception:
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data)

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data)
        print(self.actor.eval(), self.critic.eval())