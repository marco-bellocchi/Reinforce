#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 08:56:04 2024

@author: marco
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import copy
from rl_base import BaseControlAgent
from rl_tictactoe import RandomAgent, Environment, Board

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.dl1 = nn.Linear(9, 36)
        self.dl2 = nn.Linear(36, 36)
        self.output_layer = nn.Linear(36, 9)
        

    def forward(self, x):
        x = self.dl1(x)
        x = torch.relu(x)

        x = self.dl2(x)
        x = torch.relu(x)

        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QLearningApproximateControl(BaseControlAgent):
    
    def __init__(self, epsilon = 0.1, alpha = 0.1, discount_factor = 1, identifier = 1, initial_state_action_value = 0.2):
       super().__init__(epsilon, alpha, discount_factor)
       self._last_action = None
       self._last_state = None
       self._num_episodes = 1
       self.identifier = identifier
       self._initial_state_action_value = initial_state_action_value
       self._model = Linear_QNet(9, 128, 9)
       self._optimizer = optim.Adam(self._model.parameters(), lr=self._alpha)
       self._fixed_nn = copy.deepcopy( self._model)
       # self._target = copy.deepcopy( self._model)
       # self._target_optimizer = optim.Adam(self._target.parameters(), lr=self._alpha)
       self._criterion = nn.MSELoss()
       self.games = 1
       
    def take_action(self, board):
        all_actions = board.get_possible_actions()
        if np.random.uniform() < self._epsilon/self._num_episodes:
            return np.random.choice(all_actions)
        else:
            current_state = torch.tensor(board.current_state, dtype=torch.float)
            prediction = self._model(current_state)
            ordered_best_actions = torch.topk(prediction.flatten(), 9).indices
            for tensor_action in ordered_best_actions:
                action = tensor_action.item()
                if action in all_actions:
                    return action
        print("Issue here!")
        
    def first_step(self, board):
        action = self.take_action(board)
        self._last_action = action
        self._last_state = board.current_state
        return action
    
    def _nn_step(self, board, reward, terminated):
        # last_action = [0]*9
        # last_action[self._last_action] = 1
        state = torch.tensor(self._last_state, dtype=torch.float)
        if board is not None:
            next_state = torch.tensor(board.current_state, dtype=torch.float)
            next_state = torch.unsqueeze(next_state, 0)
        action = torch.tensor(self._last_action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        # if len(state.shape) == 1:
        state = torch.unsqueeze(state, 0)
        # next_state = torch.unsqueeze(next_state, 0)
        action = torch.unsqueeze(action, 0)
        reward = torch.unsqueeze(reward, 0)

        # 1: predicted Q values with current state
        pred = self._model(state)#
        
        #use it to calculate target
        # self._fixed_nn = copy.deepcopy( self._model)
        # self._fixed_optimizer = optim.Adam(self._target.parameters(), lr=self._alpha)

        target = pred.clone()
        for idx in range(len(state)):
            Q_new = reward[idx]
            if not terminated:
                Q_new = reward[idx] + self._discount_factor * torch.max(self._fixed_nn(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
            
        self._optimizer.zero_grad()
        loss = self._criterion(target, pred)
        loss.backward()
        self._optimizer.step()

        if self.games % 20 == 0:
            # print("ecco ", self.games)
            self._fixed_nn = copy.deepcopy(self._model)
            
    def step(self, board, reward):
        
        self._nn_step(board, reward, False)
        action = self.take_action(board)
        self._last_action = action
        self._last_state = board.current_state
        return action
                    
    def last_step(self, board, reward):
        self._nn_step(board, reward, True)
        self._num_episodes += 0.01
        self.games += 1
        self._last_action = None
        self._last_state = None
        
board1 = Board()
agent_1 = QLearningApproximateControl(alpha = 0.2, identifier = 1)
agent_2 = RandomAgent(identifier = -1)
env1 = Environment(board1, agent_1, agent_2, True)

while True:
    for i in range(1000):
        env1.run_episode()
        env1.reset()
        
    num_episodes = 0
    num_wins = 0
    num_losses = 0
    num_draws = 0
    for i in range(10000):
        env1.play()
        num_episodes +=1
        if board1.winner == agent_1.identifier:
            num_wins += 1
        elif board1.winner == agent_2.identifier:
            num_losses += 1
        else:
            num_draws += 1
        env1.reset()
    print(f"Statistics: Not losses:{(num_wins+num_draws)/num_episodes}, Losses:{num_losses/num_episodes}")