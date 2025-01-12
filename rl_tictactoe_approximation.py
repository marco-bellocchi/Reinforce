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
from tqdm import tqdm
import matplotlib.pyplot as plt
from rl_base import BaseControlAgent
from rl_tictactoe import RandomAgent, Environment, Board, QLearningControl

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
    
    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))
        self.eval()

class QLearningApproximateControl(BaseControlAgent):
    
    def __init__(self, epsilon = 0.1, alpha = 0.1, discount_factor = 1, identifier = 1, initial_state_action_value = 0.2, use_target_network = True, tau=0.01):
       super().__init__(epsilon, alpha, discount_factor)
       self._last_action = None
       self._last_state = None
       self._num_episodes = 1
       self.identifier = identifier
       self._initial_state_action_value = initial_state_action_value
       self._model = Linear_QNet(9, 256, 9)
       self._optimizer = optim.Adam(self._model.parameters())
       self._target_network = Linear_QNet(9, 256, 9)
       self._tau = tau
       self._criterion = nn.MSELoss()
       self._steps = 1
       self._use_target_network = use_target_network
       
    def take_action(self, board):
        all_actions = board.get_possible_actions()
        if np.random.uniform() < self._epsilon/self._num_episodes:
            return np.random.choice(all_actions)
        else:
            current_state = torch.tensor(board.current_state, dtype=torch.float)
            current_state = torch.unsqueeze(current_state, 0)
            # print(current_state)
            prediction = self._model(current_state)
            # print(prediction)
            ordered_best_actions = torch.topk(prediction.flatten(), 9).indices
            for tensor_action in ordered_best_actions:
                action = tensor_action.item()
                if action in all_actions:
                    # print(action)
                    return action
        print("Issue here!")
        
    def first_step(self, board):
        action = self.take_action(board)
        self._last_action = action
        self._last_state = board.current_state
        return action
    
    def _nn_step(self, board, reward, terminated):
        self._steps += 1
        state = torch.tensor(self._last_state, dtype=torch.float)
        state = torch.unsqueeze(state, 0)
        if board is not None:
            next_state = torch.tensor(board.current_state, dtype=torch.float)
            next_state = torch.unsqueeze(next_state, 0)
        reward = torch.tensor(reward, dtype=torch.float)
        pred = self._model(state)#
        target = pred.clone()
        Q_new = reward
        target_network = self._model
        if self._use_target_network and self._steps % 20000 == 0:
            print("updating target net")
            target_net_state_dict = self._target_network.state_dict()
            policy_net_state_dict = self._model.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self._tau + target_net_state_dict[key]*(1-self._tau)
            self._target_network.load_state_dict(target_net_state_dict)
            target_network = self._target_network
            
        if not terminated:
            Q_new = reward + self._discount_factor * torch.max(target_network(next_state[0]))
        target[0][self._last_action] = Q_new
            
        self._optimizer.zero_grad()
        loss = self._criterion(target, pred)
        loss.backward()
        self._optimizer.step()
            
    def step(self, board, reward):
        
        self._nn_step(board, reward, False)
        action = self.take_action(board)
        self._last_action = action
        self._last_state = board.current_state
        return action
                    
    def last_step(self, board, reward):
        self._nn_step(board, reward, True)
        self._num_episodes += 0.01
        self._last_action = None
        self._last_state = None

def train():
    board1 = Board()
    agent_1 = QLearningApproximateControl(alpha = 0.1, identifier = 1)
    # agent_1._model.load()
    # agent_1 = QLearningApproximateControl(alpha = 0.05, identifier = 1)
    agent_2 = RandomAgent(identifier = -1)
    env1 = Environment(board1, agent_1, agent_2, True)
    best = 0
    while True:
        for i in range(500):
            env1.run_episode()
            env1.reset()  
        num_episodes = 0
        num_wins = 0
        num_losses = 0
        num_draws = 0
        for i in range(1000):
            env1.play()
            num_episodes +=1
            if board1.winner == agent_1.identifier:
                num_wins += 1
            elif board1.winner == agent_2.identifier:
                num_losses += 1
            else:
                num_draws += 1
            env1.reset()
        print(f"Statistics: Wins:{(num_wins)/num_episodes}, Not losses:{(num_wins+num_draws)/num_episodes}, Losses:{num_losses/num_episodes}")
        # if (num_wins+num_draws)/num_episodes > best:
        #     best =(num_wins+num_draws)/num_episodes
        #     agent_1._model.save()
# train()
num_runs = 20
num_ep = 1000
scores_replay = np.zeros((num_runs, num_ep))

for run in tqdm(range(num_runs)):           # tqdm is what creates the progress bar below
    np.random.seed(run)
    board1 = Board()
    agent_1 = QLearningApproximateControl(alpha = 0.1, identifier = 1)
    # agent_1._model.load()
    # agent_1 = QLearningApproximateControl(alpha = 0.05, identifier = 1)
    agent_2 = RandomAgent(identifier = -1)
    env1 = Environment(board1, agent_1, agent_2, False)
    
    for episode_number in range(num_ep):
        env1.run_episode()
        env1.reset()
        num_wins = 0
        num_losses = 0
        num_draws = 0
        num_episodes = 0
        for _ in range(100):
            env1.play()
            num_episodes +=1
            if board1.winner == agent_1.identifier:
                num_wins += 1
            elif board1.winner == agent_2.identifier:
                num_losses += 1
            else:
                num_draws += 1
            env1.reset()
        scores_replay[run,episode_number] = (num_wins+num_draws)/num_episodes

scores_no_replay = np.zeros((num_runs, num_ep))

for run in tqdm(range(num_runs)):           # tqdm is what creates the progress bar below
    np.random.seed(run)
    board1 = Board()
    agent_1 = QLearningApproximateControl(alpha = 0.1, identifier = 1)
    # agent_1._model.load()
    # agent_1 = QLearningApproximateControl(alpha = 0.05, identifier = 1)
    agent_2 = RandomAgent(identifier = -1)
    env1 = Environment(board1, agent_1, agent_2, True)
    
    for episode_number in range(num_ep):
        env1.run_episode()
        env1.reset()
        num_wins = 0
        num_losses = 0
        num_draws = 0
        num_episodes = 0
        for _ in range(100):
            env1.play()
            num_episodes +=1
            if board1.winner == agent_1.identifier:
                num_wins += 1
            elif board1.winner == agent_2.identifier:
                num_losses += 1
            else:
                num_draws += 1
            env1.reset()
        scores_no_replay[run,episode_number] = (num_wins+num_draws)/num_episodes

scores_tabular_q = np.zeros((num_runs, num_ep))
for run in tqdm(range(num_runs)):           # tqdm is what creates the progress bar below
    np.random.seed(run)
    board1 = Board()
    agent_1 = QLearningControl(alpha = 0.27, identifier = 1)
    # agent_1._model.load()
    # agent_1 = QLearningApproximateControl(alpha = 0.05, identifier = 1)
    agent_2 = RandomAgent(identifier = -1)
    env1 = Environment(board1, agent_1, agent_2, True)
    
    for episode_number in range(num_ep):
        env1.run_episode()
        env1.reset()
        num_wins = 0
        num_losses = 0
        num_draws = 0
        num_episodes = 0
        for _ in range(100):
            env1.play()
            num_episodes +=1
            if board1.winner == agent_1.identifier:
                num_wins += 1
            elif board1.winner == agent_2.identifier:
                num_losses += 1
            else:
                num_draws += 1
            env1.reset()
        scores_tabular_q[run,episode_number] = (num_wins+num_draws)/num_episodes

scores_tabular_q_no_replay = np.zeros((num_runs, num_ep))
for run in tqdm(range(num_runs)):           # tqdm is what creates the progress bar below
    np.random.seed(run)
    board1 = Board()
    agent_1 = QLearningControl(alpha = 0.27, identifier = 1)
    # agent_1._model.load()
    # agent_1 = QLearningApproximateControl(alpha = 0.05, identifier = 1)
    agent_2 = RandomAgent(identifier = -1)
    env1 = Environment(board1, agent_1, agent_2, False)
    
    for episode_number in range(num_ep):
        env1.run_episode()
        env1.reset()
        num_wins = 0
        num_losses = 0
        num_draws = 0
        num_episodes = 0
        for _ in range(100):
            env1.play()
            num_episodes +=1
            if board1.winner == agent_1.identifier:
                num_wins += 1
            elif board1.winner == agent_2.identifier:
                num_losses += 1
            else:
                num_draws += 1
            env1.reset()
        scores_tabular_q_no_replay[run,episode_number] = (num_wins+num_draws)/num_episodes
            
        
# scores_no_replay = np.array(scores)
average_scores_replay = np.mean(scores_replay, axis=0)
average_scores_no_replay = np.mean(scores_no_replay, axis=0)
averaget_scores_tabular_q = np.mean(scores_tabular_q, axis=0)
averaget_scores_tabular_q_no_replay = np.mean(scores_tabular_q_no_replay, axis=0)
plt.figure(figsize=(15, 5), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(average_scores_replay)
plt.plot(average_scores_no_replay)
plt.plot(averaget_scores_tabular_q)
plt.plot(averaget_scores_tabular_q_no_replay)
plt.legend(["Avg. Replay", "Avg. No Replay", "Avg. Tabular Q", "Avg. Tabular Q NR"])
plt.title("Average Score per learning episode")
plt.xlabel("Steps")
plt.ylabel("Average % of non loosing games")
plt.show()