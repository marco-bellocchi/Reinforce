#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:29:55 2024

@author: marco
"""
import numpy as np
from rl_base import BaseEnvironment, BasePredictionAgent, BaseControlAgent, Experience
# np.random.seed(42)

class Board:
    
    def __init__(self, initial_state= (0,0,0,0,0,0,0,0,0)):
        self._initial_state = initial_state
        self._current_state = tuple(self._initial_state)
        self._winner = 0
    
    @property
    def current_state(self):
        return self._current_state
    
    @property
    def winner(self):
        return self._winner
    
    def get_possible_actions(self):
        free_moves = []
        for index, value in enumerate(self._current_state):
            if value == 0:
                free_moves.append(index)
        return free_moves
    
    def move(self, player, index):

        free_moves = self.get_possible_actions()
        next_state = list(self._current_state)
        if index in free_moves:
            next_state[index] = player
        else:
            raise Exception("Invalid move")
        self._current_state = tuple(next_state)
        # self.print()
        return self
    
    def is_end_game(self):
        if abs(sum(self.current_state[0:3])) == 3:
            self._winner = self.current_state[0]
            return (True, self.current_state[0])
        if abs(sum(self.current_state[3:6])) == 3:
            self._winner = self.current_state[3]
            return (True, self.current_state[3])
        if abs(sum(self.current_state[6:9])) == 3:
            self._winner = self.current_state[6]
            return (True, self.current_state[6])
        
        if abs(sum(self.current_state[0:9:3])) == 3:
            self._winner = self.current_state[0]
            return (True, self.current_state[0])
        if abs(sum(self.current_state[1:9:3])) == 3:
            self._winner = self.current_state[1]
            return (True, self.current_state[1])
        if abs(sum(self.current_state[2:9:3])) == 3:
            self._winner = self.current_state[2]
            return (True, self.current_state[2])
        
        if abs(sum(self.current_state[0:9:4])) == 3:
            self._winner = self.current_state[0]
            return (True, self.current_state[0])
        if abs(sum(self.current_state[2:8:2])) == 3:
            self._winner = self.current_state[2]
            return (True, self.current_state[2])
        
        if len(self.get_possible_actions()) == 0:
            return (True, 0)
        
        return (False, 0)

    def reset(self):
        self._current_state = tuple(self._initial_state)
        self._winner = 0
        
    def print(self):
        print("----------")
        print(f"{self.current_state[0]: < 2} {self.current_state[1]: < 2} {self.current_state[2]: < 2}")
        print(f"{self.current_state[3]: < 2} {self.current_state[4]: < 2} {self.current_state[5]: < 2}")
        print(f"{self.current_state[6]: < 2} {self.current_state[7]: < 2} {self.current_state[8]: < 2}")
        print("----------")

class RandomAgent:
    
    def __init__(self, identifier):
        self.identifier = identifier
    
    def take_action(self, board):
        possible_actions = board.get_possible_actions()
        return np.random.choice(possible_actions)

    def first_step(self, board):
        return self.take_action(board)
    
    def step(self, board, reward):
        return self.take_action(board)
    
    def last_step(self, board, reward):
        pass

class QLearningControl(BaseControlAgent):
    
    def __init__(self, epsilon = 0.1, alpha = 0.1, discount_factor = 1, identifier = 1, initial_state_action_value = 0.2):
       super().__init__(epsilon, alpha, discount_factor)
       self._last_action = None
       self._last_state = None
       self._num_episodes = 1
       self.identifier = identifier
       self._initial_state_action_value = initial_state_action_value
       
      
    def _init_q_value(self, board):
        board_state = board.current_state
        if board_state not in self._q_value:
            possible_actions = board.get_possible_actions()
            self._q_value[board_state] = {}
            for action in possible_actions:
                self._q_value[board_state][action] = self._initial_state_action_value
    ##Use max value from python!
    def _max_value(self, board):
        self._init_q_value(board)
        action_values = self._q_value[board.current_state]
        max_action_value = -1
        for action, value in action_values.items():
            if value > max_action_value:
                max_action_value = value
        return max_action_value
               
    def take_action(self, board):
        self._init_q_value(board)
        action_values = self._q_value[board.current_state]
        max_action_value = -1
        best_actions = []
        all_actions = []
        for action, value in action_values.items():
            all_actions.append(action)
            if value > max_action_value:
                best_actions = []
                best_actions.append(action)
                max_action_value = value
            elif value == max_action_value:
                best_actions.append(action)
        if np.random.uniform() < self._epsilon/self._num_episodes:
            return np.random.choice(all_actions)
        else:
            return np.random.choice(best_actions)
        
    
    def first_step(self, board):
        action = self.take_action(board)
        self._last_action = action
        self._last_state = board.current_state
        return action
    
    def step(self, board, reward):
        new_q_value = self.value(self._last_state, 
                                 self._last_action) + self._alpha*(reward + self._discount_factor *
                                                                  self._max_value(board) - self.value(self._last_state, 
                                                                                                              self._last_action))
        self.update_qvalue(self._last_state, self._last_action, new_q_value)
        action = self.take_action(board)
        self._last_action = action
        self._last_state = board.current_state
        return action
                    
    def last_step(self, board, reward):
        new_q_value = self.value(self._last_state, self._last_action) + self._alpha*(reward - self.value(self._last_state, 
                                                                                                        self._last_action))
        self.update_qvalue(self._last_state, self._last_action, new_q_value)
        self._num_episodes += 0.01
        self._last_action = None
        self._last_state = None
        
class Environment(BaseEnvironment):
    
    def __init__(self, board, agent_1, agent_2, experience_replay_active = False):
        super().__init__(board, agent_1)
        self.agent_2 = agent_2
        self._experience = Experience()
        self._experience_replay_active = experience_replay_active
    
    def _take_action(self, action, player):
        new_state = self.current_state.move(player, action)
        end_game, winner = new_state.is_end_game()
        if end_game:
            if self.current_state.winner == self._agent.identifier:
                return (1, True, new_state)
            elif self.current_state.winner == self.agent_2.identifier:
                return (-1, True, new_state)
            else:
                return (0, True, new_state)
        else:
            return (0, False, new_state)
        
        
    def run_episode(self):
        terminated = False
        action = self._agent.first_step(self.current_state)
        reward, terminated, self._current_state = self._take_action(action, self._agent.identifier)
        action = self.agent_2.first_step(self.current_state)
        reward, terminated, self._current_state = self._take_action(action, self.agent_2.identifier)
        while not terminated:
            self._experience.append(self._agent._last_state, self._agent._last_action, self.current_state.current_state, reward)
            action = self._agent.step(self.current_state, reward)
            reward, terminated, self._current_state = self._take_action(action, self._agent.identifier)
            if not terminated:
                action = self.agent_2.first_step(self.current_state)
                reward, terminated, self._current_state = self._take_action(action, self.agent_2.identifier)
        
        self._experience.append(self._agent._last_state, self._agent._last_action, None, reward)
        self._agent.last_step(self.current_state, reward)
        if self._experience_replay_active:
            self.replay()
            
    def replay(self):
        for i in range(10):
            state, action, next_state, reward = self._experience.get_state()
            self._agent._last_action = action
            self._agent._last_state = state
            if next_state is not None:
                self._agent.step(Board(next_state), reward)
            else:
                self._agent.last_step(None, reward)
    
    def play(self):
        action = self._agent.first_step(self.current_state)
        reward, terminated, self._current_state = self._take_action(action, self._agent.identifier)
        action = self.agent_2.first_step(self.current_state)
        reward, terminated, self._current_state = self._take_action(action, self.agent_2.identifier)
        while not terminated:
            action = self._agent.first_step(self.current_state)
            reward, terminated, self._current_state = self._take_action(action, self._agent.identifier)
            if not terminated:
                action = self.agent_2.first_step(self.current_state)
                reward, terminated, self._current_state = self._take_action(action, self.agent_2.identifier)
    
    def reset(self):
        self._current_state.reset()
                
class EnvironmentAgentSecondPlayer(BaseEnvironment):
    
    def __init__(self, board, agent_1, agent_2, experience_replay_active = False):
        super().__init__(board, agent_1)
        self.agent_2 = agent_2
        self._experience = Experience()
        self._experience_replay_active = experience_replay_active
    
    def _take_action(self, action, player):
        new_state = self.current_state.move(player, action)
        end_game, winner = new_state.is_end_game()
        if end_game:
            if self.current_state.winner == self._agent.identifier:
                return (1, True, new_state)
            elif self.current_state.winner == self.agent_2.identifier:
                return (-1, True, new_state)
            else:
                return (0, True, new_state)
        else:
            return (0, False, new_state)
        
    def run_episode(self):
        terminated = False
        
        action = self.agent_2.first_step(self.current_state)
        reward, terminated, self._current_state = self._take_action(action, self.agent_2.identifier)
        
        action = self._agent.first_step(self.current_state)
        reward, terminated, self._current_state = self._take_action(action, self._agent.identifier)
        
        while not terminated:
            action = self.agent_2.first_step(self.current_state)
            reward, terminated, self._current_state = self._take_action(action, self.agent_2.identifier)
            
            if not terminated:
                self._experience.append(self._agent._last_state, self._agent._last_action, self.current_state.current_state, reward)
                action = self._agent.step(self.current_state, reward)
                reward, terminated, self._current_state = self._take_action(action, self._agent.identifier)
        
        self._experience.append(self._agent._last_state, self._agent._last_action, None, reward)
        self._agent.last_step(self.current_state, reward)
        if self._experience_replay_active:
            self.replay()
            
    def replay(self):
        for i in range(10):
            state, action, next_state, reward = self._experience.get_state()
            self._agent._last_action = action
            self._agent._last_state = state
            if next_state is not None:
                self._agent.step(Board(next_state), reward)
            else:
                self._agent.last_step(None, reward)
    
    def play(self):
        
        action = self.agent_2.first_step(self.current_state)
        reward, terminated, self._current_state = self._take_action(action, self.agent_2.identifier)
        
        action = self._agent.first_step(self.current_state)
        reward, terminated, self._current_state = self._take_action(action, self._agent.identifier)
        
        while not terminated:
            action = self.agent_2.first_step(self.current_state)
            reward, terminated, self._current_state = self._take_action(action, self.agent_2.identifier)
            
            if not terminated:
                action = self._agent.first_step(self.current_state)
                reward, terminated, self._current_state = self._take_action(action, self._agent.identifier)
                
    def reset(self):
        self._current_state.reset()

'''
board1 = Board()
agent_1 = QLearningControl(alpha = 0.27, identifier = 1)
agent_2 = RandomAgent(identifier = -1)
env1 = Environment(board1, agent_1, agent_2, False)
while True:
    for i in range(300000):
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
    break

board2 = Board()
agent_3 = QLearningControl(alpha = 0.3, identifier = -1)
agent_4 = RandomAgent(identifier = 1)
env2 = EnvironmentAgentSecondPlayer(board2, agent_3, agent_4, False)
for i in range(1000000):
    env2.run_episode()
    env2.reset()
    
num_episodes = 0
num_wins = 0
num_losses = 0
num_draws = 0
for i in range(10000):
    env2.play()
    num_episodes +=1
    if board2.winner == agent_3.identifier:
        num_wins += 1
    elif board2.winner == agent_4.identifier:
        num_losses += 1
    else:
        num_draws += 1
    env2.reset()

print(f"Statistics: Not losses:{(num_wins+num_draws)/num_episodes}, Losses:{num_losses/num_episodes}")
#Statistics: Not losses:0.9936, Losses:0.0064
'''