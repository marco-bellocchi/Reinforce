#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:56:44 2024

@author: marco
"""
import random
import inspect
from kaggle_environments import make, evaluate, utils
env = make("connectx", debug=True)
a = env.agents['negamax']
dir(a)
inspect.getsource(a.play)
a({'board':[0]*42, 'columns':7}, config = {'columns': 7})

print(list(env.agents))
    
env.render(mode="ipython")

# Selects random valid column
def agent_random(obs, config):
    print(obs)
    print(config)
    # valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    # return random.choice(valid_moves)

env.run(["random", negamax_agent])

class KaggleAgent:
    
    def __init__(self, agent):
        self.agent - agent
        
    def run(self, obs, config):
        board = Connect4Board(obs)
        return self.agent.first_step(board)
    
def negamax_agent(obs, config):
    columns = config.columns
    rows = config.rows
    size = rows * columns

    # Due to compute/time constraints the tree depth must be limited.
    max_depth = 4
    EMPTY = 0
    def negamax(board, mark, depth):
        
        moves = sum(1 if cell != EMPTY else 0 for cell in board)

        # Tie Game
        if moves == size:
            return (0, None)

        # Can win next.
        for column in range(columns):
            if board[column] == EMPTY and is_win(board, column, mark, config, False):
                return ((size + 1 - moves) / 2, column)

        # Recursively check all columns.
        best_score = -size
        best_column = None
        for column in range(columns):
            if board[column] == EMPTY:
                # Max depth reached. Score based on cell proximity for a clustering effect.
                if depth <= 0:
                    row = max(
                        [
                            r
                            for r in range(rows)
                            if board[column + (r * columns)] == EMPTY
                        ]
                    )
                    score = (size + 1 - moves) / 2
                    if column > 0 and board[row * columns + column - 1] == mark:
                        score += 1
                    if (
                        column < columns - 1
                        and board[row * columns + column + 1] == mark
                    ):
                        score += 1
                    if row > 0 and board[(row - 1) * columns + column] == mark:
                        score += 1
                    if (
                        row < rows - 2
                        and board[(row + 1) * columns + column] == mark
                    ):
                        score += 1
                else:
                    next_board = board[:]
                    play(next_board, column, mark, config)
                    (score, _) = negamax(next_board, 1 if mark == 2 else 2, depth - 1)
                    score = score * -1
                if score > best_score or (score == best_score and choice([True, False])):
                    best_score = score
                    best_column = column

        return (best_score, best_column)

    _, column = negamax(obs.board[:], obs.mark, max_depth)
    if column == None:
        column = choice([c for c in range(columns) if obs.board[c] == EMPTY])
    return column