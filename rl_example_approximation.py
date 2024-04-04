#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:57:15 2024

@author: marco
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import copy
from rl_base import BaseControlAgent, softmax
from rl_examples import Environment
# from rl_tictactoe import RandomAgent, Environment, Board


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QLearningApproximateControl(BaseControlAgent):
    
    def __init__(self, epsilon = 0.1, alpha = 0.1, discount_factor = 1):
       super().__init__(epsilon, alpha, discount_factor)
       self._last_action = None
       self._last_state = None
       self._model = Linear_QNet(1, 16, 2)
       self._optimizer = optim.Adam(self._model.parameters())
       self._fixed_nn = copy.deepcopy(self._model)
       self._criterion = nn.MSELoss()
       self.games = 1
       self._num_episodes = 1
       
    def take_action(self, current_state):
        if np.random.uniform() < self._epsilon:
            return np.random.choice([-1,1])
        current_state = torch.tensor(current_state, dtype=torch.float)
        current_state = torch.unsqueeze(current_state, 0)
        prediction = self._model(current_state)
        action_values_probs = prediction.detach().numpy()
        left_action_value = action_values_probs[0]
        right_action_value = action_values_probs[1]
        if left_action_value > right_action_value:
            return -1
        elif left_action_value < right_action_value:
            return 1
        else:
            return np.random.choice([-1,1])
        # softmax_input = np.array([action_values_probs])
        # return np.random.choice([-1,1], p=softmax(softmax_input))
        
    def first_step(self, current_state):
        action = self.take_action(current_state)
        self._last_action = action
        self._last_state = current_state
        return action
    
    def _nn_step(self, current_state, reward, terminated):
        state_tensor = torch.tensor(self._last_state, dtype=torch.float).unsqueeze(0)
        next_state_tensor = None
        if not terminated:
            next_state_tensor = torch.tensor(current_state, dtype=torch.float).unsqueeze(0)
        reward_tensor = torch.tensor(reward, dtype=torch.float).unsqueeze(0)

        pred = self._model(state_tensor)

        target = pred.clone()
        idx = 0
        Q_new = reward_tensor[idx]
        # print(Q_new)
        if not terminated:
            # next_pred = self._fixed_nn(next_state_tensor)
            Q_new = reward_tensor[idx] + self._discount_factor * torch.max(self._model(next_state_tensor))
        target_index_action = 0
        if self._last_action == 1:
            target_index_action = 1
        target[target_index_action] = Q_new

        self._optimizer.zero_grad()
        # print(target, pred)
        loss = self._criterion(target, pred)
        loss.backward()
        self._optimizer.step()

        # if self.games % 4 == 0:
        #     # print("ecco ", self.games)
        #     self._fixed_nn = copy.deepcopy(self._model)
            
    def step(self, current_state, reward):
        
        self._nn_step(current_state, reward, False)
        action = self.take_action(current_state)
        self._last_action = action
        self._last_state = current_state
        return action
                    
    def last_step(self, current_state, reward):
        self._nn_step(current_state, reward, True)
        self._num_episodes += 0.01
        self.games += 1
        self._last_action = None
        self._last_state = None
        
agent = QLearningApproximateControl(alpha = 0.2)
env = Environment(3, agent, False)

while True:
    for i in range(2):
        env.run_episode()
        env.reset()
    
    win = 0
    loss = 0
    for i in range(1000):
        # env._printing = True
        env.play()
        if env.current_state == 0:
            loss+=1
        elif env.current_state == 6:
            win+=1
        env.reset()
    
    print(win, loss)
