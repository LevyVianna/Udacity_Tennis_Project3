import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 3141

class FCNetwork(nn.Module):
    
    def __init__(self, state_size, output_size, hidden_size_1, hidden_size_2, is_actor, action_size = 2, num_agents = 2):
        super(FCNetwork, self).__init__()
        
        self.seed = torch.manual_seed(SEED)
        
        if is_actor:
            self.linear1 = nn.Linear(state_size, hidden_size_1)
            self.output_gate = F.tanh
        else:
            self.linear1 = nn.Linear((state_size+action_size)*num_agents, hidden_size_1)
            self.output_gate = None
            
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, output_size)
        self.reset_parameters()

    def reset_parameters(self):
        u_range = 1. / np.sqrt(self.linear1.weight.data.size()[0])
        self.linear1.weight.data.uniform_(-u_range, u_range)
        u_range = 1. / np.sqrt(self.linear2.weight.data.size()[0])
        self.linear2.weight.data.uniform_(-u_range, u_range)
        self.linear3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, x, actions=None):
        if actions is not None:
            x = torch.cat((x, actions.float()), dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        if self.output_gate:
            x = self.output_gate(x)
        return x
