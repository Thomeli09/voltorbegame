"""Regression tests for optional solver methods and legacy policy."""

import numpy as np
import pytest

from voltorbe_game import PuzzleClues, VoltorbGame, legacy_scores, solve
from voltorbe_game.simulation import example_game, main


def small_game() -> VoltorbGame:
    return VoltorbGame(PuzzleClues(
        row_points=[2, 2], row_voltorbs=[1, 1],
        column_points=[2, 2], column_voltorbs=[1, 1],
    ))


def test_sample_methods_select_different_safe_tiles():
    exact = solve(example_game(), method='exact')
    legacy = solve(example_game(), method='legacy')
    assert exact.method == 'exact' and legacy.method == 'legacy'
    assert exact.mode == legacy.mode == 'exact'
    assert exact.possible_grids == legacy.possible_grids == 828
    np.testing.assert_allclose(exact.probabilities, legacy.probabilities)
    assert (exact.recommendation.row, exact.recommendation.column) == (1, 4)
    assert (legacy.recommendation.row, legacy.recommendation.column) == (0, 4)
    assert exact.recommendation.voltorb_risk == legacy.recommendation.voltorb_risk == 0
    assert legacy.strategy == 'legacy_score'


def test_legacy_scores_keep_infinite_safe_lines_and_revealed_zero():
    game = example_game()
    scores = legacy_scores(game)
    assert scores.shape == (5, 5)
    assert np.all(np.isinf(scores[:, 4]))  # Column five has no Voltorbs
    game.reveal(1, 4, 2)
    scores = legacy_scores(game)
    assert scores[1, 4] == 0


def test_legacy_fallback_is_explicit_and_does_not_claim_probabilities():
    result = solve(small_game(), method='legacy', max_nodes=1, fallback=True)
    assert result.method == 'legacy'
    assert result.mode == 'heuristic'
    assert result.strategy == 'legacy_score'
    assert result.possible_grids is None
    assert result.probabilities is None
    assert result.multiplier_chance is None
    assert result.recommendation is not None


def test_direct_heuristic_skips_enumeration():
    result = solve(small_game(), method='heuristic', max_nodes=1)
    assert result.mode == 'heuristic'
    assert result.method == 'heuristic'
    assert result.visited_nodes == 0
    assert result.possible_grids is None
    assert result.recommendation is not None


def test_recommendation_excludes_revealed_tile():
    game = small_game()
    game.reveal(0, 0, 2)
    for method in ('exact', 'legacy', 'heuristic'):
        result = solve(game, method=method)
        assert result.recommendation is not None
        assert (result.recommendation.row, result.recommendation.column) != (0, 0)


def test_won_or_lost_game_not_recommended():
    game = small_game()
    game.reveal(0, 0, 2)
    game.reveal(1, 1, 2)
    for method in ('exact', 'legacy', 'heuristic'):
        assert solve(game, method=method).recommendation is None


def test_rejects_invalid_solver_strategy_combinations():
    game = small_game()
    with pytest.raises(ValueError, match='method'):
        solve(game, method='not_an_engine')
    with pytest.raises(ValueError, match='strategy'):
        solve(game, method='legacy', strategy='risk_first')
    with pytest.raises(ValueError, match='strategy'):
        solve(game, method='exact', strategy='legacy_score')


def test_legacy_cli_once(capsys):
    assert main(['--once', '--method', 'legacy']) == 0
    assert 'Solver: legacy' in capsys.readouterr().out


def test_heuristic_cli_once(capsys):
    assert main(['--once', '--method', 'heuristic']) == 0
    assert 'Solver: heuristic' in capsys.readouterr().out
