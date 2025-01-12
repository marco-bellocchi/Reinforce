#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:46:47 2024

@author: marco
"""
import numpy as np
import time

import pickle
from rl_base import BaseEnvironment, BasePredictionAgent, BaseControlAgent, PrioritisedExperience, Experience
# np.random.seed(42)

# Python code to demonstrate namedtuple()
from collections import namedtuple

# Declaring namedtuple()
Configuration = namedtuple('Configuration', ['columns', 'rows', 'inarow'])
Observation = namedtuple('Configuration', ['board', 'mark'])
from kaggle_environments import make, evaluate, utils

env = make("connectx", debug=True)
# a = env.agents['negamax']
# config = Configuration(7,6,4)
# b = [0]*42
# obs = Observation(b, 2)
# a(obs, config)

class KAgent:
    
    def __init__(self, identifier, k_agent):
        self.identifier = identifier
        self.k_agent = env.agents['negamax']
        self.config = Configuration(7,6,4)
    
    def take_action(self, board):
        # possible_actions = board.get_possible_actions()
        obs = Observation(list(board.current_state), self.identifier)
        return self.k_agent(obs, self.config)

    def first_step(self, board):
        return self.take_action(board)
    
    def step(self, board, reward):
        return self.take_action(board)
    
    def last_step(self, board, reward):
        pass

k_agent = KAgent(2, env.agents['negamax'])

class Connect4Board:
    
    def __init__(self, initial_state=[0] * 42):
        self._initial_state = initial_state
        self._current_state = list(self._initial_state)
        self._winner = 0
    
    @property
    def current_state(self):
        return tuple(self._current_state)
    
    @property
    def winner(self):
        return self._winner
    
    def get_possible_actions(self):
        free_moves = []
        for col in range(7):
            for row in range(6):
                if self._current_state[row * 7 + col] == 0:
                    free_moves.append(col)
                    break
        return free_moves
    
    def move(self, player, column):
        free_moves = self.get_possible_actions()
        if column not in free_moves:
            raise Exception("Invalid move")
        
        for row in range(5, -1, -1):
            if self._current_state[row * 7 + column] == 0:
                self._current_state[row * 7 + column] = player
                break
                
        return self
    
    def get_winning_move(self, player):
        free_moves = self.get_possible_actions()
        for column in range(7):
           if column not in free_moves:
               continue
           for row in range(5, -1, -1):
               if self._current_state[row * 7 + column] == 0:
                   self._current_state[row * 7 + column] = player
                   if self.is_end_game():
                       self._current_state[row * 7 + column] = 0
                       return column
                   self._current_state[row * 7 + column] = 0
        return -1 
    
    def get_num_of_steps(self):
        return len([vol for vol in self._current_state if vol != 0])
    
    def is_end_game(self):
        # Check rows
        for row in range(6):
            for col in range(4):
                val = self._current_state[row * 7 + col]
                if val != 0:
                    if all(val == self._current_state[row * 7 + col + i] for i in range(1, 4)):
                        self._winner = val
                        return True, val
        
        # Check columns
        for col in range(7):
            for row in range(3):
                val = self._current_state[row * 7 + col]
                if val != 0:
                    if all(val == self._current_state[(row + i) * 7 + col] for i in range(1, 4)):
                        self._winner = val
                        return True, val
        
        # Check diagonals
        for row in range(3):
            for col in range(4):
                val = self._current_state[row * 7 + col]
                if val != 0:
                    if all(val == self._current_state[(row + i) * 7 + col + i] for i in range(1, 4)):
                        self._winner = val
                        return True, val
                    
        for row in range(3):
            for col in range(3, 7):
                val = self._current_state[row * 7 + col]
                if val != 0:
                    if all(val == self._current_state[(row + i) * 7 + col - i] for i in range(1, 4)):
                        self._winner = val
                        return True, val
        
        if 0 not in self._current_state:
            return True, 0
        
        return False, 0

    def reset(self):
        self._current_state = list(self._initial_state)
        self._winner = 0
        
    def print(self):
        # print("  0  1  2  3  4  5  6")
        for row in range(6):
            res = "  ".join(str(self._current_state[row * 7 + col]) for col in range(7)).replace("-1", "2")
            print(res)


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
    
class RandomAgentSmarter:
    
    def __init__(self, identifier):
        self.identifier = identifier
    
    def take_action(self, board):
        possible_actions = board.get_possible_actions()
        win_move = board.get_winning_move(self.identifier)
        if win_move!= -1:
            return win_move
        return np.random.choice(possible_actions)

    def first_step(self, board):
        return self.take_action(board)
    
    def step(self, board, reward):
        return self.take_action(board)
    
    def last_step(self, board, reward):
        pass

class QLearningControl(BaseControlAgent):
    
    def __init__(self, epsilon = 0.1, alpha = 0.1, discount_factor = 1, identifier = 1, initial_state_action_value = .5):
       super().__init__(epsilon, alpha, discount_factor)
       self._last_action = None
       self._last_td_error = 0
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
        max_action_value = -100000
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
        td_error = (reward + self._discount_factor *
                                         self._max_value(board) - self.value(self._last_state,  self._last_action))
        new_q_value = self.value(self._last_state, 
                                 self._last_action) + self._alpha*(td_error)
        self.update_qvalue(self._last_state, self._last_action, new_q_value)
        action = self.take_action(board)
        self._last_action = action
        self._last_state = board.current_state
        self._last_td_error = td_error
        return action
                    
    def last_step(self, board, reward):
        td_error = reward - self.value(self._last_state, self._last_action)
        new_q_value = self.value(self._last_state, self._last_action) + self._alpha*(td_error)
        self.update_qvalue(self._last_state, self._last_action, new_q_value)
        self._num_episodes += 0.0001
        self._last_action = None
        self._last_state = None
        self._last_td_error = td_error
        
class Environment(BaseEnvironment):
    
    def __init__(self, board, agent_1, agent_2, experience_replay_active = False):
        super().__init__(board, agent_1)
        self.agent_2 = agent_2
        self._experience = PrioritisedExperience()
        self._experience_replay_active = experience_replay_active
    
    def _take_action(self, action, player):
        new_state = self.current_state.move(player, action)
        end_game, winner = new_state.is_end_game()
        # print("------------------")
        # new_state.print()
        # print("------------------")
        if end_game:
            # print("end_game!")
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
        # self._current_state.print()
        # print("-------------")
        # time.sleep(3)
        action = self.agent_2.first_step(self.current_state)
        reward, terminated, self._current_state = self._take_action(action, self.agent_2.identifier)
        # self._current_state.print()
        # print("-------------")
        # time.sleep(3)
        while not terminated:
            self._experience.append(self._agent._last_state, self._agent._last_action, self.current_state.current_state, reward, self._agent._last_td_error)
            action = self._agent.step(self.current_state, reward)
            reward, terminated, self._current_state = self._take_action(action, self._agent.identifier)
            # self._current_state.print()
            # print("-------------")
            # time.sleep(3)
            if not terminated:
                action = self.agent_2.first_step(self.current_state)
                reward, terminated, self._current_state = self._take_action(action, self.agent_2.identifier)
                # self._current_state.print()
                # print("-------------")
                # time.sleep(3)
        
        self._experience.append(self._agent._last_state, self._agent._last_action, None, reward, self._agent._last_td_error)
        self._agent.last_step(self.current_state, reward)
        if self._experience_replay_active:
            self.replay()
            
    def replay(self):
        for i in range(300):
            state, action, next_state, reward = self._experience.get_state()
            if state is not None:
                self._agent._last_action = action
                self._agent._last_state = state
                if next_state is not None:
                    self._agent.step(Connect4Board(next_state), reward)
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
        self._experience = PrioritisedExperience()
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
                self._experience.append(self._agent._last_state, self._agent._last_action, self.current_state.current_state, reward, self._agent._last_td_error)
                action = self._agent.step(self.current_state, reward)
                reward, terminated, self._current_state = self._take_action(action, self._agent.identifier)
        
        self._experience.append(self._agent._last_state, self._agent._last_action, None, reward, self._agent._last_td_error)
        self._agent.last_step(self.current_state, reward)
        if self._experience_replay_active:
            self.replay()
            
    def replay(self):
        for i in range(10):
            state, action, next_state, reward = self._experience.get_state()
            self._agent._last_action = action
            self._agent._last_state = state
            if next_state is not None:
                self._agent.step(Connect4Board(next_state), reward)
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
                
board1 = Connect4Board()
agent_1 = QLearningControl(alpha = 0.05, identifier = 1)#0.05
agent_1._num_episodes = 1
agent_2 = k_agent#RandomAgent(identifier = 2)
env1 = Environment(board1, agent_1, agent_2, True)
while True:
    num_episodes = 0
    num_wins = 0
    num_losses = 0
    num_draws = 0
    tot_steps = 0
    for i in range(100):
        env1.run_episode()
        tot_steps += env1.current_state.get_num_of_steps()
        num_episodes +=1
        if board1.winner == agent_1.identifier:
            num_wins += 1
        elif board1.winner == agent_2.identifier:
            num_losses += 1
        else:
            num_draws += 1
        env1.reset()
    print(f"Statistics: Wins:{(num_wins)/num_episodes}, Not losses:{(num_wins+num_draws)/num_episodes}, Losses:{num_losses/num_episodes}, Avg steps: {tot_steps/num_episodes}")


board11 = Connect4Board()
agent_11 = QLearningControl(alpha = 0.05, identifier = -1)#0.05
agent_22 = RandomAgent(identifier = 1)
env11 = EnvironmentAgentSecondPlayer(board11, agent_11, agent_22, False)
while True:
    for i in range(10000):
        env11.run_episode()
        # board11.print()
        # print(board11.winner)
        env11.reset()
        # board11.print()
        
    
    num_episodes = 0
    num_wins = 0
    num_losses = 0
    num_draws = 0
    for i in range(100):
        env11.play()
        num_episodes +=1
        if board11.winner == agent_11.identifier:
            num_wins += 1
        elif board11.winner == agent_22.identifier:
            num_losses += 1
        else:
            num_draws += 1
        env11.reset()
    print(f"Statistics: Wins:{(num_wins)/num_episodes}, Not losses:{(num_wins+num_draws)/num_episodes}, Losses:{num_losses/num_episodes}")
    
# agent_1._num_episodes = 1

a = agent_1._q_value

with open('negmax.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('negmax.pickle', 'rb') as handle:
    b = pickle.load(handle)
agent_1._q_value = b
# print(a == b)

def user_play(env, trained_agent):
    print("Play please:")
    index = int(input())
    while index not in env._current_state.get_possible_actions():
        print("Not valid move, try again")
        index = int(input())
    return env._take_action(index, -1*trained_agent.identifier) 

def play_interactive(env, trained_agent):
    trained_agent._last_action = None
    trained_agent._last_status = None
    terminated = False
    env.reset()
    action = trained_agent.first_step(env.current_state)
    reward, terminated, env._current_state = env._take_action(action, trained_agent.identifier)
    env._current_state.print()
    print("-------------------------")
    reward, terminated, env._current_state = user_play(env, trained_agent)
    env._current_state.print()
    print("-------------------------")
    while not terminated:
        action = trained_agent.step(env.current_state, reward)
        reward, terminated, env._current_state = env._take_action(action, trained_agent.identifier)
        env._current_state.print()
        print("-------------------------")
        if not terminated:
            reward, terminated, env._current_state = user_play(env, trained_agent)
            env._current_state.print()
            print("-------------------------")
    trained_agent.last_step(env.current_state, reward)
    print(f"Game ended, the winner is {env._current_state.winner}")


# def play_interactive(env, trained_agent):
#     # trained_agent._epsilon = 0
#     trained_agent._last_action = None
#     trained_agent._last_status = None
#     terminated = False
#     env.reset()
#     action = trained_agent.first_step(env._current_state)
#     reward, terminated, env._current_state = env._take_action(action, trained_agent.identifier)
#     env._current_state.print()
#     print("-------------------------")
#     while not terminated:
#         print("Play please")
#         index = int(input())
#         while index not in env._current_state.get_possible_actions():
#             print("Not valid move, try again")
#             index = int(input())
#         reward, terminated, env._current_state = env._take_action(index, -1*trained_agent.identifier)
#         env._current_state.print()
#         print("-------------------------")
#         if terminated:
#             trained_agent.last_step(env._current_state, reward)
#             break
#         action = trained_agent.step(env._current_state, reward)
#         reward, terminated, env._current_state = env._take_action(action, trained_agent.identifier)
#         env._current_state.print()
#         print("-------------------------")
#         trained_agent.last_step(env._current_state, reward)
        
#     print(f"Game ended, the winner is {env._current_state.winner}")

play_interactive(env1, agent_1)

#Play 2 RL agent agaist each other to learn more
agent_2 = agent_11
agent_22 = agent_1
while True:
    agent_2 = agent_11
    board1 = Connect4Board()
    env1 = Environment(board1, agent_1, agent_2, False)
    
    for i in range(10000):
        env1.run_episode()
        env1.reset()
    num_episodes = 0
    num_wins = 0
    num_losses = 0
    num_draws = 0
    for i in range(100):
        env1.play()
        num_episodes +=1
        if board1.winner == agent_1.identifier:
            num_wins += 1
        elif board1.winner == agent_2.identifier:
            num_losses += 1
        else:
            num_draws += 1
        env1.reset()
    print(f"Statistics first player training: Wins:{(num_wins)/num_episodes}, Not losses:{(num_wins+num_draws)/num_episodes}, Losses:{num_losses/num_episodes}")
    agent_22 = agent_1
    board11 = Connect4Board()
    env11 = EnvironmentAgentSecondPlayer(board11, agent_11, agent_22, False)
    for i in range(10000):
        env11.run_episode()
        env11.reset()
    num_episodes = 0
    num_wins = 0
    num_losses = 0
    num_draws = 0
    for i in range(100):
        env11.play()
        num_episodes +=1
        if board11.winner == agent_11.identifier:
            num_wins += 1
        elif board11.winner == agent_22.identifier:
            num_losses += 1
        else:
            num_draws += 1
        env11.reset()
    print(f"Statistics second player: Wins:{(num_wins)/num_episodes}, Not losses:{(num_wins+num_draws)/num_episodes}, Losses:{num_losses/num_episodes}")
    
agent_1.boards = []
agent_1.actions = []
def agent_rl(obs, config):
    board = Connect4Board(obs['board'])
    agent_1.boards.append(board)
    to_return = int(agent_1.first_step(board))
    agent_1.actions.append(to_return)
    return to_return

def negamax_agent(obs, configs):
    board = Connect4Board(obs['board'])
    agent_1.boards.append(board)
    to_return = int(k_agent.first_step(board))
    return to_return

def get_win_percentages(agent1, agent2, n_rounds):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time          
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)
    # Agent 2 goes first (roughly) half the time      
    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent1, agent2], config, [], n_rounds-n_rounds//2)]
    print("Negamax Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))
    print("Our Agent  Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))
    print("Number of Invalid Plays by Our Agent", outcomes.count([None, 0]))
    print("Number of Invalid Plays by Negamaxur Agent :", outcomes.count([0, None]))
    
get_win_percentages(agent_rl, negamax_agent, 100)
for b in agent_1.boards:
    print("--------------")
    b.print()