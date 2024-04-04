#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:56:47 2024

@author: marco
"""

import numpy as np
from rl_base import BaseEnvironment, BasePredictionAgent, BaseControlAgent


class Environment(BaseEnvironment):
    
    def __init__(self, state, agent, printing = False):
        super().__init__(state, agent)
        self._printing = printing
    
    def _take_action(self, action):
        self._current_state += action
        if self._current_state == 0:
            return (0, True, self._current_state)
        if self._current_state == 6:
            return (1, True, self._current_state)
        return (0, False, self._current_state)
    
    def print(self, action):
        if self._printing:
            print("----")
            print(self._current_state)
            print(action)
            print("----")
        
    def run_episode(self):
        terminated = False
        action = self._agent.first_step(self._current_state)
        self.print(action)
        reward, terminated, self._current_state = self._take_action(action)
        while not terminated:
            action = self._agent.step(self._current_state, reward)
            self.print(action)
            reward, terminated, self._current_state = self._take_action(action)
        self._agent.last_step(self._current_state, reward)
    
    def play(self):
        terminated = False
        while not terminated:
            action = self._agent.first_step(self._current_state)
            self.print(action)
            reward, terminated, self._current_state = self._take_action(action)
            
    
    def reset(self):
        self._current_state = self._original_state
        
class TDAgentPrediction(BasePredictionAgent):
    
    def __init__(self, epsilon = 0.1, alpha = 0.1, discount_factor = 1):
       super().__init__(epsilon, alpha, discount_factor)
       self._last_action = None
       self._last_state = None
       self._actions = [-1,1]
       for state in range(7):
           self._v_value[state] = 0
    
    def take_action(self, current_state):
        return np.random.choice(self._actions)
    
    def first_step(self, current_state):
        action = self.take_action(current_state)
        self._last_action = action
        self._last_state = current_state
        return action
    
    def step(self, current_state, reward):
        new_v_value = self.value(self._last_state) + self._alpha*(reward + self._discount_factor *
                                                                  self.value(current_state) - self.value(self._last_state))
        self.update_value(self._last_state, new_v_value)
        action = self.take_action(current_state)
        self._last_action = action
        self._last_state = current_state
        return action
                    
    def last_step(self, current_state, reward):
        new_v_value = self.value(self._last_state) + self._alpha*(reward - self.value(self._last_state))
        self.update_value(self._last_state, new_v_value)
        self._last_action = None
        self._last_state = None
        
agent = TDAgentPrediction(alpha=0.05)
env = Environment(3, agent)
for i in range(1000):
    env.run_episode()
    env.reset()
agent._v_value

class QLearningControl(BaseControlAgent):
    
    def __init__(self, epsilon = 0.1, alpha = 0.1, discount_factor = 1):
       super().__init__(epsilon, alpha, discount_factor)
       self._last_action = None
       self._last_state = None
       self._actions = [-1,1]
       for state in range(7):
           self._q_value[state] = {-1:0, 1:0}
           
    def _max_value(self, state):
        action_values = self._q_value[state]
        left_value = action_values[-1]
        right_value = action_values[1]
        if left_value > right_value:
            return left_value
        return right_value
    
    def take_action(self, current_state):
        action_values = self._q_value[current_state]
        left_value = action_values[-1]
        right_value = action_values[1]
        if np.random.uniform() < self._epsilon:
            return np.random.choice(self._actions)
        elif left_value > right_value:
            return -1
        elif left_value < right_value:
            return 1
        else:
            return np.random.choice([-1,1])
    
    def first_step(self, current_state):
        action = self.take_action(current_state)
        self._last_action = action
        self._last_state = current_state
        return action

    def step(self, current_state, reward):
        new_q_value = self.value(self._last_state, 
                                 self._last_action) + self._alpha*(reward + self._discount_factor *
                                                                  self._max_value(current_state) - self.value(self._last_state, 
                                                                                                              self._last_action))
        self.update_qvalue(self._last_state, self._last_action, new_q_value)
        action = self.take_action(current_state)
        self._last_action = action
        self._last_state = current_state
        return action
                    
    def last_step(self, current_state, reward):
        new_q_value = self.value(self._last_state, self._last_action) + self._alpha*(reward - self.value(self._last_state, 
                                                                                                        self._last_action))
        self.update_qvalue(self._last_state, self._last_action, new_q_value)
        self._last_action = None
        self._last_state = None
        
agent = QLearningControl(alpha = 0.05)
env = Environment(3, agent)
for i in range(1000):
    env.run_episode()
    env.reset()
agent._q_value

# win = 0
# loss = 0
# for i in range(10000):
# # env._printing = False
#     env.play()
#     if env.current_state == 0:
#         loss+=1
#     elif env.current_state == 6:
#         win+=1
#     env.reset()

# print(win, loss)

class MCAgentPrediction(BasePredictionAgent):
    
    def __init__(self, epsilon = 0.1, alpha = 0.1, discount_factor = 1):
       super().__init__(epsilon, alpha, discount_factor)
       self._returns = []
       self._actions = [-1,1]
       self.episode_states = []
       self.episode_actions = []
       self.episode_returns = []
       for state in range(7):
           self._v_value[state] = 0
           self._returns.append([])
    
    def take_action(self, current_state):
        return np.random.choice(self._actions)
    
    def first_step(self, current_state):
        action = self.take_action(current_state)
        self.episode_actions.append(action)
        self.episode_states.append(current_state)
        return action
    
    def step(self, current_state, reward):
        self.episode_returns.append(reward)
        action = self.take_action(current_state)
        self.episode_actions.append(action)
        self.episode_states.append(current_state)
        return action
    
    def last_step(self, current_state, reward):
        self.episode_returns.append(reward)
        g_return = 0
        t = len(self.episode_returns) - 1
        while t > -1:
            g_return = self._discount_factor * g_return + self.episode_returns[t]
            s_t = self.episode_states[t]
            if not s_t in self.episode_states[0:t]:
                self._returns[s_t].append(g_return)
                self.update_value(s_t, np.average(self._returns[s_t]))
            t += -1
        self.episode_states = []
        self.episode_actions = []
        self.episode_returns = [] 
           
agent = MCAgentPrediction(alpha=0.05)
env = Environment(3, agent)
for i in range(10000):
    env.run_episode()
    env.reset()
agent._v_value

class MCAgentControl(BaseControlAgent):
    
    def __init__(self, epsilon = 0.1, alpha = 0.1, discount_factor = 1):
       super().__init__(epsilon, alpha, discount_factor)
       self._returns = {}
       self._policy = {}
       self._actions = [-1,1]
       self.episode_states_actions = []
       self.episode_returns = []
       for state in range(1,6):
           for action in self._actions :
               state_action = f"{state}_{action}"
               self._policy[state_action] = 0.5
               self._q_value[state_action] = 0
               self._returns[state_action] = []
    
    def value(self, state, action):
        state_action = f"{state}_{action}"
        return self._q_value[state_action]
    
    def update_value(self, state, action, value):
        state_action = f"{state}_{action}"
        self._q_value[state_action] = value
        
    def _get_best_action(self, current_state):
        best_action = None
        left_value = self.value(current_state, -1)
        right_value = self.value(current_state, 1)
        if left_value > right_value:
            best_action = -1
        elif right_value > left_value:
            best_action = 1
        else:
            best_action = np.random.choice(self._actions)
        return best_action
    
    def _update_policy(self, current_state):
        num_actions = len(self._actions)
        best_action = self._get_best_action(current_state)
        for action in self._actions:
            state_action = f"{current_state}_{action}"
            if action == best_action:
                self._policy[state_action] = 1 -self._epsilon + self._epsilon/num_actions
            else:
                self._policy[state_action] = self._epsilon/num_actions 
    
    def take_action(self, current_state):
        actions = []
        probs = []
        for action in self._actions:
            state_action = f"{current_state}_{action}"
            actions.append(action)
            probs.append(self._policy[state_action])
        return np.random.choice(actions, p = probs)
    
    def first_step(self, current_state):
        action = self.take_action(current_state)
        self.episode_states_actions.append((current_state, action))
        return action
    
    def step(self, current_state, reward):
        self.episode_returns.append(reward)
        action = self.take_action(current_state)
        self.episode_states_actions.append((current_state, action))
        return action
    
    def last_step(self, current_state, reward):
        self.episode_returns.append(reward)
        g_return = 0
        t = len(self.episode_returns) - 1
        while t > -1:
            g_return = self._discount_factor * g_return + self.episode_returns[t]
            s_t, a_t = self.episode_states_actions[t]
            state_action = f"{s_t}_{a_t}" 
            if not (s_t, a_t) in self.episode_states_actions[0:t]:
                self._returns[state_action].append(g_return)
                self.update_value(s_t, a_t, np.average(self._returns[state_action]))
                self._update_policy(s_t)
            t += -1
        self.episode_returns = []
        self.episode_states_actions = []

agent = MCAgentControl(alpha=0.05)
env = Environment(3, agent)
for i in range(1000):
    env.run_episode()
    env.reset()
