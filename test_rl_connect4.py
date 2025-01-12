import pytest
from rl_connect4 import Connect4Board

@pytest.fixture
def board():
    return Connect4Board()

def test_initial_state(board):
    initial_state = [0] * 42
    assert board.current_state == initial_state

def test_get_possible_actions(board):
    possible_actions = [0, 1, 2, 3, 4, 5, 6]
    assert board.get_possible_actions() == possible_actions

def test_move(board):
    board.move(1, 0)
    assert board.current_state[35] == 1

def test_is_end_game_no_winner(board):
    assert not board.is_end_game()[0]

def test_is_end_game_winner(board):
    # Horizontal win
    board.move(1, 0)
    board.move(1, 1)
    board.move(1, 2)
    board.move(1, 3)
    assert board.is_end_game()[0]
    assert board.is_end_game()[1] == 1

    # Vertical win
    board.reset()
    board.move(1, 0)
    board.move(1, 0)
    board.move(1, 0)
    board.move(1, 0)
    assert board.is_end_game()[0]
    assert board.is_end_game()[1] == 1

    # Diagonal win
    board.reset()
    board.move(2, 0)
    board.move(2, 1)
    board.move(2, 1)
    board.move(1, 2)
    board.move(2, 2)
    board.move(2, 2)
    board.move(2, 2)
    board.move(1, 3)
    board.move(2, 3)
    board.move(2, 3)
    board.move(2, 3)
    board.move(1, 3)
    board.move(2, 3)
    board.print()
    assert board.is_end_game()[0]
    assert board.is_end_game()[1] == 2

def test_reset(board):
    board.move(1, 0)
    board.reset()
    initial_state = [0] * 42
    assert board.current_state == initial_state

def test_diagonal_win_positive(board):
    # Diagonal win from bottom-left to top-right
    board.move(1, 0)
    board.move(2, 1)
    board.move(1, 1)
    board.move(2, 2)
    board.move(2, 2)
    board.move(1, 2)
    board.move(2, 3)
    board.move(1, 2)
    board.move(1, 3)
    board.move(2, 3)
    board.move(1, 3)
    assert board.is_end_game()[0]
    assert board.is_end_game()[1] == 1

def test_horizontal_win_positive(board):
    # Horizontal win
    board.reset()
    board.move(1, 0)
    board.move(2, 0)
    board.move(1, 1)
    board.move(2, 1)
    board.move(1, 2)
    board.move(2, 2)
    board.move(1, 3)
    assert board.is_end_game()[0]
    assert board.is_end_game()[1] == 1

def test_vertical_win_positive(board):
    # Vertical win
    board.reset()
    board.move(1, 0)
    board.move(1, 1)
    board.move(1, 2)
    board.move(1, 3)
    assert board.is_end_game()[0]
    assert board.is_end_game()[1] == 1

def test_no_winner(board):
    # No winner
    board.reset()
    assert not board.is_end_game()[0]

def test_invalid_move(board):
    # Invalid move
    board.reset()
    with pytest.raises(Exception):
        board.move(1, 10)
