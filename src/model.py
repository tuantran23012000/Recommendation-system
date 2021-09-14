import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, action_sequence_length):
        super(Critic, self).__init__()
        self.encode_state = nn.LSTM(state_size,action_size,batch_first = True)
        hidden_stack = [nn.Linear((action_sequence_length + 1)*action_size, hidden_size),
                             nn.ReLU(),]
        for i in range(3):
            hidden_stack.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.hidden_layer = nn.Sequential(*hidden_stack)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        if not isinstance(state,torch.Tensor):
            state = torch.tensor(state)
        if not isinstance(action,torch.Tensor):
            action = torch.tensor(action)
        if (len(state.shape)==2) and (len(action.shape)==2):
            action = action.unsqueeze(0)
            state = state.unsqueeze(0)
        _,(encoded_state,__) = self.encode_state(state)
        encoded_state = encoded_state.squeeze(0)
        action = action.flatten(1)
        x = torch.cat([encoded_state,action],-1)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        if (len(state.shape)==2) and (len(action.shape)==2):
            x = x.squeeze(0)
        return x

class Actor(nn.Module):
    def __init__(self, input_size,input_sequence_length, output_sequence_length, output_size):
        super(Actor, self).__init__()
        self.weight_matrix = torch.nn.Parameter(torch.ones((1,input_sequence_length), requires_grad=True))
        self.Linear = nn.Linear(input_size, output_size)
        self.Activation = nn.Softmax(dim=-1)
        self.output_shape = (output_sequence_length,output_size)
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        state = torch.FloatTensor(state)
        size = len(state.shape)
        if size==2:
            state = state.unsqueeze(0)
        state = self.weight_matrix.matmul(state)
        state = state.squeeze(1)
        action = []
#        x = self.Linear(state)
        action.append(self.Activation(state))
        for i in range(self.output_shape[0]-1):
            indices = action[i].argmax(-1).unsqueeze(-1)
            action_i = action[i].scatter(-1,indices,0)
            action_i = action_i / action_i.sum(-1).unsqueeze(-1)
            action.append(action_i)
        action = torch.cat(action,-1).reshape((-1,self.output_shape[0],self.output_shape[1]))
        if size==2:
            action = action.squeeze(0)
        return action