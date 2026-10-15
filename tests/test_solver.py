"""Exact enumeration, risk ranking and fallbacks."""

import numpy as np
import pytest

from voltorbe_game import (
    InconsistentPuzzleError, PuzzleClues, SearchLimitExceeded,
    VoltorbGame, heuristic_scores, solve, valid_line_candidates,
)
from voltorbe_game.simulation import example_game


def small_game() -> VoltorbGame:
    return VoltorbGame(PuzzleClues(
        row_points=[2, 2], row_voltorbs=[1, 1],
        column_points=[2, 2], column_voltorbs=[1, 1],
    ))


def test_valid_line_candidates_respect_known():
    assert valid_line_candidates(2, 1, [-1, -1]) == [(0, 2), (2, 0)]
    assert valid_line_candidates(2, 1, [2, -1]) == [(2, 0)]


def test_exact_solver_two_symmetrical_boards():
    result = solve(small_game())
    assert result.possible_grids == 2
    assert result.mode == 'exact'
    np.testing.assert_allclose(result.voltorb_risk, 0.5)
    np.testing.assert_allclose(result.multiplier_chance, 0.5)
    np.testing.assert_allclose(result.probabilities.sum(axis=2), 1.0)
    assert result.recommendation is not None


def test_reveal_filters_possible_boards():
    game = small_game()
    game.reveal(0, 0, 2)
    result = solve(game)
    assert result.possible_grids == 1
    assert result.recommendation is not None
    assert (result.recommendation.row, result.recommendation.column) == (1, 1)
    assert result.recommendation.voltorb_risk == 0.0
    assert result.multiplier_chance[1, 1] == 1
    assert np.isnan(result.voltorb_risk[0, 0])


def test_completed_game_no_recommendation():
    game = small_game()
    game.reveal(0, 0, 2)
    game.reveal(1, 1, 2)
    assert solve(game).recommendation is None


def test_inconsistent_board_raises():
    game = VoltorbGame(PuzzleClues(
        row_points=[2, 2], row_voltorbs=[1, 1],
        column_points=[3, 1], column_voltorbs=[1, 1],
    ))
    with pytest.raises(InconsistentPuzzleError):
        solve(game)


def test_search_budget_is_explicit():
    with pytest.raises(SearchLimitExceeded):
        solve(small_game(), max_nodes=1)
    fallback = solve(small_game(), max_nodes=1, fallback=True)
    assert fallback.mode == 'heuristic'
    assert fallback.probabilities is None
    assert fallback.possible_grids is None


def test_sample_from_original():
    result = solve(example_game())
    assert result.possible_grids == 828
    assert result.visited_nodes > 0
    assert result.recommendation is not None
    assert result.recommendation.voltorb_risk == 0.0
    assert result.recommendation.multiplier_chance > 0.0


def test_heuristic_is_not_a_calibrated_probability():
    risk, scores = heuristic_scores(example_game())
    assert risk.shape == (5, 5)
    assert scores.shape == (5, 5)
    assert np.all(scores >= 0)


def test_expected_gain_option():
    result = solve(example_game(), strategy='expected_gain')
    assert result.strategy == 'expected_gain'
    assert result.recommendation is not None


def test_invalid_strategy():
    with pytest.raises(ValueError):
        solve(small_game(), strategy='other')


def test_exact_count_matches_bruteforce_small_rectangular():
    """Independent brute-force check on a 2-row, 3-column board."""
    from itertools import product

    known_board = np.array([[3, 0, 1], [1, 2, 0]])
    clues = PuzzleClues(
        row_points=known_board.sum(axis=1).tolist(),
        row_voltorbs=(known_board == 0).sum(axis=1).tolist(),
        column_points=known_board.sum(axis=0).tolist(),
        column_voltorbs=(known_board == 0).sum(axis=0).tolist(),
    )
    game = VoltorbGame(clues)
    result = solve(game)
    possibilities = []
    for flat in product(range(4), repeat=6):
        matrix = np.array(flat).reshape(2, 3)
        if (np.array_equal(matrix.sum(axis=1), clues.row_points)
                and np.array_equal(matrix.sum(axis=0), clues.column_points)
                and np.array_equal((matrix == 0).sum(axis=1), clues.row_voltorbs)
                and np.array_equal((matrix == 0).sum(axis=0), clues.column_voltorbs)):
            possibilities.append(matrix)
    assert result.possible_grids == len(possibilities)
    assert len(possibilities) >= 1
    brute_risk = (np.stack(possibilities) == 0).mean(axis=0)
    np.testing.assert_allclose(result.voltorb_risk, brute_risk)


def test_score_with_larger_maximum():
    clues = PuzzleClues(
        row_points=[4], row_voltorbs=[0],
        column_points=[4], column_voltorbs=[0], max_value=4,
    )
    game = VoltorbGame(clues)
    result = solve(game)
    assert result.scores[0, 0] == 4
    assert result.multiplier_chance[0, 0] == 1
