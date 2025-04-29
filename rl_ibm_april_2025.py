#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 12:02:02 2025

@author: marco
"""

import math
import numpy as np
from rl_base import BaseEnvironment, BasePredictionAgent, BaseControlAgent, Experience

class Board:
    def __init__(self, current_position=(0, 0), current_size =1, num_of_moves = 0, initial_state=[]):
     
        self._current_position = current_position
        self._current_size = current_size
        self._num_of_moves = num_of_moves
        self._initial_state = np.array(initial_state)
        self._current_state = np.array(initial_state)
        self._list_actions = []
        self._max_moves = 4000
        self.rows, self.columns = self._current_state.shape

    @property
    def current_position(self):
        return self._current_position

    @property
    def current_size(self):
        return self._current_size

    @property
    def current_state(self):
        return self._current_state

    def sum_of_neighbors(self):
        eating_capacity = self.calculate_eating_capacity()
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
        
        sum_neighbors = 0
        x, y = self.current_position
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.columns:
                if eating_capacity >= self.current_state[nx][ny]:
                    sum_neighbors += self.current_state[nx][ny]
        return sum_neighbors

    def get_possible_actions(self):
        free_moves = []
        x, y = self.current_position
        capacity = self.calculate_eating_capacity()
        
        if x + 1 < self.rows and self.current_state[x + 1,y] <= capacity:
            free_moves.append("D")
        if x - 1 >= 0 and self.current_state[x - 1, y] <= capacity:
            free_moves.append("U")
        if y + 1 < self.columns and self.current_state[x, y + 1] <= capacity:
            free_moves.append("R")
        if y - 1 >= 0 and self.current_state[x, y - 1] <= capacity:
            free_moves.append("L")
        
        return free_moves

    def calculate_eating_capacity(self):
        return int(math.sqrt(self.current_size))

    def move(self, move):
        x, y = self._current_position
        self._list_actions.append(move)
        zeroed_col_row = False
    
        if move == "U":
            value = self._current_state[x-1, y]
            self._current_position = (x-1, y)
            self._current_size += value
            self._current_state[x-1, y] = 0
            if sum(self._current_state[x-1, :]) == 0 or sum(self._current_state[:, y]) == 0:
                zeroed_col_row = True
    
        elif move == "D":
            value = self._current_state[x+1, y]
            self._current_position = (x+1, y)
            self._current_size += value
            self._current_state[x+1, y] = 0
            if sum(self._current_state[x+1, :]) == 0 or sum(self._current_state[:, y]) == 0:
                zeroed_col_row = True
    
        elif move == "R":
            value = self._current_state[x, y+1]
            self._current_position = (x, y+1)
            self._current_size += value
            self._current_state[x, y+1] = 0
            if sum(self._current_state[x, :]) == 0 or sum(self._current_state[:, y+1]) == 0:
                zeroed_col_row = True
    
        elif move == "L":
            value = self._current_state[x, y-1]
            self._current_position = (x, y-1)
            self._current_size += value
            self._current_state[x, y-1] = 0
            if sum(self._current_state[x, :]) == 0 or sum(self._current_state[:, y-1]) == 0:
                zeroed_col_row = True
    
        self._num_of_moves += 1
        return self, zeroed_col_row

    def is_end_game(self):
        return self._num_of_moves > self._max_moves or self._current_size == 3273
    
    def won(self):
        return self._current_size == 3273
        
    def reset(self):
        self._list_actions = []
        self._current_position = (0, 0)
        self._num_of_moves = 0
        self._current_size = 1
        self._current_state = np.array(self._initial_state)

class Environment(BaseEnvironment):
    
    def __init__(self, board, agent_1):
        super().__init__(board, agent_1)
    
    def _take_action(self, action, player):
        old_size = self._current_state.current_size
        new_state, zeroed_row_col = self._current_state.move(action)
        new_size = self._current_state.current_size
        diff_size = new_size - old_size
        if new_state.is_end_game():
            if self._current_state.won():
                return (10000-self.current_state._num_of_moves)*(2000/self.current_state._num_of_moves), True, new_state
            else:
                return np.count_nonzero(self.current_state.current_state==0), True, new_state
        else:
            if diff_size > 0:
                reward = diff_size * 3
                return reward, False, new_state
            return -1, False, new_state
        
    def run_episode(self):
        terminated = False
        action = self._agent.first_step(self.current_state)
        reward, terminated, self._current_state = self._take_action(action, self._agent.identifier)

        while not terminated:
            action = self._agent.step(self.current_state, reward)
            reward, terminated, self._current_state = self._take_action(action, self._agent.identifier)
        
        self._agent.last_step(self.current_state, reward)

    def reset(self):
        self._current_state.reset()

class QLearningControl(BaseControlAgent):
    
    def __init__(self, epsilon = 0.1, alpha = 0.1, discount_factor = 1, initial_state_action_value = 0.2):
       super().__init__(epsilon, alpha, discount_factor)
       self._last_action = None
       self._last_state = None
       self._num_episodes = 1
       self.identifier = 0
       self._initial_state_action_value = initial_state_action_value
       
    def get_q_state(self, board):
        return (board.current_position[0],
                board.current_position[1],
                board.current_size,
                board.sum_of_neighbors())

        ##Use max value from python!
    def _max_value(self, board):
        board_state = self.get_q_state(board)
        if board_state not in self._q_value:
            possible_actions = board.get_possible_actions()
            self._q_value[board_state] = {}
            for action in possible_actions:
                self._q_value[board_state][action] = self._initial_state_action_value
        action_values = self._q_value[board_state]
        max_action_value = -1
        for action, value in action_values.items():
            if value > max_action_value:
                max_action_value = value
        return max_action_value
               
        
    def take_action(self, board):
        
        board_state = self.get_q_state(board)
        if board_state not in self._q_value:
            possible_actions = board.get_possible_actions()
            self._q_value[board_state] = {}
            for action in possible_actions:
                self._q_value[board_state][action] = self._initial_state_action_value
                
        action_values = self._q_value[board_state]
        max_action_value = -1000000
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
        self._last_state = self.get_q_state(board)
        return action
    
    def step(self, board, reward):
        new_q_value = self.value(self._last_state, 
                                 self._last_action) + self._alpha*(reward + self._discount_factor *
                                                                  self._max_value(board) - self.value(self._last_state, 
                                                                                                              self._last_action))
        self.update_qvalue(self._last_state, self._last_action, new_q_value)
        action = self.take_action(board)
        self._last_action = action
        self._last_state = self.get_q_state(board)
        return action

                            
    def last_step(self, board, reward):
        new_q_value = self.value(self._last_state, self._last_action) + self._alpha*(reward - self.value(self._last_state, 
                                                                                                        self._last_action))
        self.update_qvalue(self._last_state, self._last_action, new_q_value)
        self._num_episodes += 0.0001
        self._last_action = None
        self._last_state = None


def precompute_q_table(agent, new_board):
    if agent._last_state not in agent._q_value:
        possible_actions = new_board.get_possible_actions()
        agent._q_value[agent._last_state] = {}
        for action in possible_actions:
            print(action)
            agent._q_value[agent._last_state][action] = agent._initial_state_action_value
            
def reinforce_trajectory(new_board, agent, env, list_actions, alpha = 0.5):
    first_action = list_actions[0]
    agent._last_action = first_action
    agent._last_state = agent.get_q_state(new_board)
    precompute_q_table(agent, new_board)
    reward, terminated, next_state = env._take_action(first_action, agent.identifier)
    for a in list_actions[1:]:
        new_q_value =  agent.value(agent._last_state, agent._last_action) + alpha*(reward + agent._discount_factor * agent._max_value(next_state) - agent.value(agent._last_state, agent._last_action))
        agent.update_qvalue(agent._last_state, agent._last_action, new_q_value)
        agent._last_action = a
        agent._last_state = agent.get_q_state(new_board)
        precompute_q_table(agent, new_board)
        reward, terminated, next_state = env._take_action(a, agent.identifier)   
        
    agent.last_step(next_state, reward)
    
import heapq

class PrioritizedQueue:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.heap = []
        self.set = set()

    def add(self, solution):
        # Negative length for max-heap behavior
        if solution in self.set:
            return
        self.set.add(solution)
        entry = (-len(solution), solution)
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, entry)
        else:
            # Only add if new solution is shorter than the longest in queue
            if -self.heap[0][0] > len(solution):
                heapq.heappop(self.heap)
                heapq.heappush(self.heap, entry)

    def __iter__(self):
        # sorted by increasing length
        sorted_solutions = sorted(self.heap, key=lambda x: len(x[1]))
        for _, solution in sorted_solutions:
            yield solution
            
    def min_len(self):
        if not self.heap:
            return None
        return min(len(sol) for _, sol in self.heap)

    def max_len(self):
        if not self.heap:
            return None
        return max(len(sol) for _, sol in self.heap)
# # Example usage:
# pq = PrioritizedQueue(max_size=100)
# pq.add([1, 2, 3])
# pq.add([1, 2])
# pq.add([1, 2, 3, 4])

# print(pq.get_sorted_solutions())


init_state = """
 0  0  1  5  1  0  0  0  0  0  1 19  0  7  4  2  7 12  7  1
 2  6  0  0  1  8  0  8  1  2  1  0  8  9  1  7 10 13 10  6
 4 11  6  7  5  5 14  1 12  1  0  2  0  2  2  5  1 10  0 14
15 12  2  5 18  6 19 16 18 11 14  3  1  2  3  3  8  2  1  9
 5  6  8 18  4 17  7 16 14 13  4 13  8  1  2  2  7  5 11 12
 6  7 13 16  1 14  7 17 18  9 14  6 16 10  0  3  2  0  6  5
11  5 11  3 14 19 19  4 17 16  3 12 17 17  1  2 12  6  7 11
18  6  6  3 19 13  7  9  5 13  4  4  2 13  2  0  0  5  4  6
17 19  7  2  4  3  4  1 16  9 13 17 17 15  6  9  1  5  2  0
 8  8 17 18 10 12 10  0  0 13 13 10  8  0  0  7 18 10  6  3
13  3 19  3  5  9 17 16 12  2 19  9  1 17  3  0 10 11  4 19
14  5 11 13 15  6  5 10  6  1  7  3  4 15 10 10 13  4  9  7
 2 12  5  7  7 16  3  2 18 14 11 18 12 15  4  2 12 15 10  6
12  5  2 15  8  9 18  9  5  1 17 17  1  0  8  9  5  6  8 13
 9 13  5  3  9  8 18 15 10  6 12 18 11 15  2 12  6  8 12 15
14  4  2  0 13  2 18 12 16  2  4 13  0  3 16 15 15 16  7  7
 6 12  1 14  4 12  8 14 10  0 15 16 13  4  5 12  5  2 16 12
 5  5  3  0  8  0  5 16 11  4 17 13 18 17  0  9  8 16 13  6
15 13 13  5  6  7  9 15 12 18  2 12 19  4  9  5  6  8  9  3
12 10 11  2  5  8 11  7 16 12  0 14 10  5  9  0 15  4 11  3
"""
pq = PrioritizedQueue(max_size=30)
data = [[int(num) for num in line.split()] for line in init_state.strip().split("\n")]
board1 = Board(initial_state=data)
# agent_1 = QLearningApproximateControl(alpha = 0.1, identifier = 1)
agent_1 = QLearningControl(epsilon = 0.03, alpha = 0.1, discount_factor=0.95, initial_state_action_value=10)#0.1, 100
env1 = Environment(board1, agent_1)#False

min_len = 3000
# sols_set = set()
found_sol = False
total_episodes = 0
while not found_sol:
    episodes = 0
    not_win = True
    while episodes < 1000:
        episodes+=1
        total_episodes+=1
        env1.run_episode()
        if board1.won():
            if len(board1._list_actions) < min_len:
                not_win = False
                min_len -=20
                if min_len < 700:
                    min_len = 700
                pq.add(tuple(board1._list_actions))
                print(min_len, len(board1._list_actions))
                # print(min_len)
                
                if len(board1._list_actions) < 501:
                    # sol = list(board1._list_actions)
                    print("Done")
                    found_sol = True
                    break
                board1._max_moves -=20
                if board1._max_moves < 700:
                    board1._max_moves = 700
            else:
                not_win +=1
        env1.reset()
    # if not_win: 
    #     agent_1._num_episodes = max(1, agent_1._num_episodes-1)
    #     print("Reducing epsilon to ", agent_1._epsilon /agent_1._num_episodes) 
    #Reinforce good paths

    print("Reinforcing good paths...")
    print(f"min len of solution is {pq.min_len()}, max len is {pq.max_len()}")
    for _ in range(25):
        for sol in pq:
            tmp_board = Board(initial_state=data)
            tmp_env = Environment(tmp_board, agent_1)
            alpha = 0.1
            if len(sol) > 3000:
                alpha = 0.2
            elif len(sol)>2000:
                alpha = 0.25
            elif len(sol)>1200:
                alpha = 0.3
            elif len(sol)>500:
                alpha = 0.2
            reinforce_trajectory(tmp_board, agent_1, tmp_env, sol, alpha)
    print("Reinforcing done!")
    if total_episodes > 10000:
        print("Resetting epsilon and pq...")
        total_episodes = 0
        agent_1._num_episodes = 1
        pq = PrioritizedQueue(max_size=30)
# agent_1._epsilon = 0.1
# # heap = pq.heap
# board1._max_moves = 900

# tot = 0
# for sol in sols_set:
#     if len(sol)<800:
#         tot+=1
min_sol = None
for sol in pq:
    print(len(sol))
    if len(sol) < 500:
        min_sol = sol
len(min_sol)

board_sol = Board(initial_state=data)    
with open("ibm_sol_steps.txt", "w") as f:
    for action in min_sol:
        f.writelines("-----------\n")
        board_sol.move(action)
        f.writelines(str(action))
        f.write("\n")
        f.writelines(str(board_sol.current_position))
        f.write("\n")
        f.writelines(str(board_sol.current_size))
        f.write("\n")
        str_arr = board_sol.current_state.astype(str)
        # Step 1: Find the maximum digit length in the array
        max_len = 3
        
        # Step 2: Convert all elements to padded strings
        str_arr = np.array([[str(x).rjust(max_len) for x in row] for row in str_arr])
        
        # Step 3: Replace the target position with 'X' (centered)
        str_arr[board_sol.current_position] = 'X'.rjust(max_len)

        for row in str_arr:
            # Join elements with spaces and write to file
            f.write(' '.join(row) + '\n')
        f.write("\n")
        f.writelines(str("-------------"))
    

with open("ibm_sol.txt", "w") as f:
    sol_str = []
    for a in min_sol:
        f.write(str(a))
    # f.write(str(sol_str))