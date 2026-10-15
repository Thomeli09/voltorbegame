"""Unit tests for puzzle clues and revealed tile invariants."""

import numpy as np
import pytest

from voltorbe_game import PuzzleClues, VoltorbGame


def small_clues() -> PuzzleClues:
    return PuzzleClues(
        row_points=[2, 2], row_voltorbs=[1, 1],
        column_points=[2, 2], column_voltorbs=[1, 1],
    )


def test_clues_dimensions_and_conversion():
    c = small_clues()
    assert c.rows == c.columns == 2
    assert c.row_points == (2, 2)


def test_clue_invalid_totals():
    with pytest.raises(ValueError, match='point totals'):
        PuzzleClues(row_points=[2, 2], row_voltorbs=[1, 1],
                    column_points=[2, 3], column_voltorbs=[1, 1])


def test_invalid_line_constraints():
    with pytest.raises(ValueError, match='Impossible'):
        PuzzleClues(row_points=[7, 1], row_voltorbs=[1, 0],
                    column_points=[4, 4], column_voltorbs=[1, 0])


def test_revealed_copy_and_rollback():
    game = VoltorbGame(small_clues())
    copy = game.revealed
    copy[0, 0] = 2
    assert game.revealed[0, 0] == -1
    game.reveal(0, 0, 2)
    with pytest.raises(ValueError):
        game.reveal(0, 1, 1)  # Contradicts row 0 (must contain one bomb)
    assert game.revealed[0, 1] == -1


def test_reveal_outside_grid():
    game = VoltorbGame(small_clues())
    with pytest.raises(IndexError):
        game.reveal(9, 0, 1)
    with pytest.raises(ValueError):
        game.reveal(0, 0, -1)


def test_finished_when_all_multipliers_are_revealed():
    game = VoltorbGame(small_clues())
    assert not game.is_won
    game.reveal(0, 0, 2)
    game.reveal(1, 1, 2)
    assert game.is_won and not game.is_lost


def test_game_is_lost_after_zero():
    game = VoltorbGame(small_clues())
    game.reveal(0, 1, 0)
    assert game.is_lost and not game.is_won


def test_board_dimensions_and_fractional_rejection():
    game = VoltorbGame(small_clues())
    with pytest.raises(ValueError):
        game.set_revealed([[1, 2, 0]])
    with pytest.raises(ValueError, match='integers'):
        game.set_revealed([[1.5, -1], [-1, -1]])
