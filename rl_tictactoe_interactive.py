#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:29:35 2024

@author: marco
"""

from rl_tictactoe import Environment, Board, QLearningControl, RandomAgent
#from rl_tictactoe_approximation import QLearningApproximateControl

def play_interactive(env, trained_agent):
    trained_agent._epsilon = 0
    trained_agent._last_action = None
    trained_agent._last_status = None
    terminated = False
    env.reset()
    while not terminated:
        action = trained_agent.first_step(env._current_state)
        reward, terminated, env._current_state = env._take_action(action, trained_agent.identifier)
        env._current_state.print()
        if terminated:
            break
        print("Play please")
        index = int(input())
        while index not in env._current_state.get_possible_actions():
            print("Not valid move, try again")
            index = int(input())
        
        reward, terminated, env._current_state = env._take_action(index, -1*trained_agent.identifier)
        env._current_state.print()
    print(f"Game ended, the winner is {env._current_state.winner}")

board1 = Board()
# agent_1 = QLearningApproximateControl(alpha = 0.1, identifier = 1)
agent_1 = QLearningControl(alpha = 0.27, identifier = 1)
agent_2 = RandomAgent(identifier = -1)
env1 = Environment(board1, agent_1, agent_2, True)
while True:
    for i in range(1000):
        env1.run_episode()
        env1.reset()
        # print(i)
    break

play_interactive(env1, agent_1)
