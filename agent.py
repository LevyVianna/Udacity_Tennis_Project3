import numpy as np
import os
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from models import *
from utils import *

BUFFER_SIZE = 50000
BATCH_SIZE = 128
GAMMA = 0.995
TAU = 0.002
LR_ACTOR = 0.0001
LR_CRITIC = 0.001
UPDATE_EVERY = 2
NUM_AGENTS = 2
SEED = 3141592

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(object):

    def __init__(self, state_size, action_size, agent_id):

        self.action_size = action_size
        self.agent_id = agent_id
        self.__name__ = 'DDPG'
        
        np.random.seed(SEED)

        # Actor Network
        self.actor_local  = FCNetwork(state_size, action_size, 400, 300, True).to(device)
        self.actor_target = FCNetwork(state_size, action_size, 400, 300, True).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network
        self.critic_local  = FCNetwork(state_size, 1, 550, 300, False, action_size, NUM_AGENTS).to(device)
        self.critic_target = FCNetwork(state_size, 1, 550, 300, False, action_size, NUM_AGENTS).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        self.noise = OUNoise(action_size, SEED)
        self.memory = ReplayBuffer(action_size, SEED, BUFFER_SIZE, BATCH_SIZE)
        self.t_step = 0

    def step(self, states, actions, reward, next_states, done):
        
        self.memory.add(states, actions, reward, next_states, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                self.learn()
                # Update Target Networks
                self.soft_update(self.critic_local, self.critic_target)
                self.soft_update(self.actor_local, self.actor_target)

    def act(self, states, noise = True, theta=0.15, sigma=0.2):
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if noise:
            actions += self.noise.sample(theta, sigma)
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self):

        (states, actions, rewards, next_states, dones) = self.memory.sample()
        
        # Organizing data from agent and his twin 
        twin_agent_id = (self.agent_id + 1) % 2
        states_ = states
        actions_ = actions
        next_states_ = next_states
        
        states = torch.from_numpy(np.vstack( np.array(states)[:,self.agent_id] )).float().to(device)
        actions = torch.from_numpy(np.vstack( np.array(actions)[:,self.agent_id] )).float().to(device)
        rewards = torch.from_numpy(np.vstack( rewards )).float().to(device)
        next_states = torch.from_numpy(np.vstack( np.array(next_states)[:,self.agent_id] )).float().to(device)
        dones = torch.from_numpy(np.vstack( dones ).astype(np.uint8)).float().to(device)
        states_twin = torch.from_numpy(np.vstack( np.array(states_)[:,twin_agent_id] )).float().to(device)
        actions_twin = torch.from_numpy(np.vstack( np.array(actions_)[:,twin_agent_id] )).float().to(device)
        next_states_twin = torch.from_numpy(np.vstack( np.array(next_states_)[:,twin_agent_id] )).float().to(device)

        # Getting Actor actions and their respective Critic values
        next_actions_both = torch.cat([self.actor_target(states), self.actor_target(states_twin)], 1).to(device)
        next_states_both = torch.cat((next_states, next_states_twin), 1).to(device)
        Q_targets_next = self.critic_target(next_states_both, next_actions_both)
        
        ## Getting Q_targets 
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

        # Critic Loss
        states_both = torch.cat((states, states_twin), 1).to(device)
        actions_both = torch.cat((actions, actions_twin), 1).to(device)
        Q_expected = self.critic_local(states_both, actions_both)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # Update Actor
        actions_pred = torch.cat((self.actor_local(states), self.actor_local(states_twin).detach()), 1).to(device)
        actor_loss = -self.critic_local(states_both, actions_pred).mean()
        
        # Minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

    def reset(self):
        self.noise.reset()