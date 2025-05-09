#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:45:35 2024

@author: marco
"""

import numpy as np
from abc import ABC, abstractmethod
from collections import deque, namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# class ReplayMemory(object):

#     def __init__(self, capacity):
#         self.memory = deque([], maxlen=capacity)

#     def push(self, *args):
#         """Save a transition"""
#         self.memory.append(Transition(*args))

#     def sample(self, batch_size):
#         return np.random.choice(self.memory)

#     def __len__(self):
#         return len(self.memory)

class PrioritisedExperience:
    
    def __init__(self, maxlen = 1000):
        self._deque_state_action = deque(maxlen = maxlen)
        self._deque_next_state_reward = deque(maxlen = maxlen)
        
    def append(self, state, action, next_state, reward, td_error):
        if abs(td_error < 0.1):
            return
        self._deque_state_action.appendleft((state, action))
        self._deque_next_state_reward.appendleft((next_state, reward, abs(td_error)))
    
    #Prioritized
    def get_state(self, batch_size = 1):
        # Assuming experience[2] is the temporal difference error
        if len(self._deque_next_state_reward) == 0:
            return None, None, None, None
        priorities = [experience[2] for experience in self._deque_next_state_reward]
        probabilities = priorities / np.sum(priorities)
        index = np.random.choice(range(len(self._deque_next_state_reward)), p=probabilities)
        state, action = self._deque_state_action[index]
        next_state, reward, td_error = self._deque_next_state_reward[index]
        return state, action, next_state, reward
    
class Experience:
    
    def __init__(self, maxlen = 10000):
        self._deque_state_action = deque(maxlen = maxlen)
        self._deque_next_state_reward = deque(maxlen = maxlen)
        
    def append(self, state, action, next_state, reward, td_error = 1):
        self._deque_state_action.appendleft((state, action))
        self._deque_next_state_reward.appendleft((next_state, reward))
        
    def get_state(self, batch_size = 1):
        index = np.random.choice(list(range(len(self._deque_state_action))))
        state, action = self._deque_state_action[index]
        next_state, reward = self._deque_next_state_reward[index]
        return state, action, next_state, reward

class BaseEnvironment(ABC):
    
    def __init__(self, state, agent):
        self._original_state = state
        self._current_state = state
        self._agent = agent
    
    @property
    def current_state(self):
        return self._current_state
    
    @abstractmethod
    def run_episode(self):
        pass

class BaseAgent(ABC):
    
    def __init__(self, epsilon = 0.1, alpha = 0.1, discount_factor = 1):
        self._epsilon = epsilon
        self._alpha = alpha
        self._discount_factor = discount_factor
        self._experience = Experience()
        
    @abstractmethod
    def take_action(self, current_state):
            pass

    @abstractmethod
    def step(self, current_state, reward):
            pass
    
    @abstractmethod
    def last_step(self, current_state, reward):
            pass
        
class BasePredictionAgent(BaseAgent):
    
    def __init__(self, epsilon = 0.1, alpha = 0.1, discount_factor = 1):
       super().__init__(epsilon, alpha, discount_factor)
       self._v_value = {}
      
    def value(self, state):
        return self._v_value[state]
    
    def update_value(self, state, value):
        self._v_value[state] = value
                
class BaseControlAgent(BaseAgent):
    
    def __init__(self, epsilon = 0.1, alpha = 0.1, discount_factor = 1):
       super().__init__(epsilon, alpha, discount_factor)
       self._q_value = {}
      
    def value(self, state, action):
        return self._q_value[state][action]
    
    def update_qvalue(self, state, action, value):
        self._q_value[state][action] = value


def softmax(action_values, tau=1.0):
    """
    Args:
        action_values (Numpy array): A 2D array of shape (batch_size, num_actions). 
                       The action-values computed by an action-value network.              
        tau (float): The temperature parameter scalar.
    Returns:
        A 2D array of shape (batch_size, num_actions). Where each column is a probability distribution over
        the actions representing the policy.
    """
    
    # Compute the preferences by dividing the action-values by the temperature parameter tau
    preferences = None
    # Compute the maximum preference across the actions
    max_preference = None
    
    # your code here
    preferences = action_values/tau
    max_preference = np.max(preferences, axis=1)

    # Reshape max_preference array which has shape [Batch,] to [Batch, 1]. This allows NumPy broadcasting 
    # when subtracting the maximum preference from the preference of each action.
    reshaped_max_preference = max_preference.reshape((-1, 1))
    
    # Compute the numerator, i.e., the exponential of the preference - the max preference.
    exp_preferences = None
    # Compute the denominator, i.e., the sum over the numerator along the actions axis.
    sum_of_exp_preferences = None
    
    # your code here
    exp_preferences = np.exp(preferences - reshaped_max_preference)
    sum_of_exp_preferences = np.sum(exp_preferences, axis = 1)
    #assert not np.isnan(exp_preferences).any()
    #assert not np.isnan(sum_of_exp_preferences).any()
    # Reshape sum_of_exp_preferences array which has shape [Batch,] to [Batch, 1] to  allow for NumPy broadcasting 
    # when dividing the numerator by the denominator.
    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))
    
    # Compute the action probabilities according to the equation in the previous cell.
    action_probs = None
    
    # your code here
    action_probs = exp_preferences/reshaped_sum_of_exp_preferences
    
    # squeeze() removes any singleton dimensions. It is used here because this function is used in the 
    # agent policy when selecting an action (for which the batch dimension is 1.) As np.random.choice is used in 
    # the agent policy and it expects 1D arrays, we need to remove this singleton batch dimension.
    action_probs = action_probs.squeeze()
    return action_probs