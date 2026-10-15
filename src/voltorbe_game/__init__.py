"""Puzzle models and constraint-based solving for Pokémon Voltorb Flip."""

from .puzzle import PuzzleClues, VoltorbGame
from .solver import (
    FlipRecommendation,
    InconsistentPuzzleError,
    SearchLimitExceeded,
    SolverResult,
    legacy_scores,
    heuristic_scores,
    SolverMethod,
    Strategy,
    solve,
    valid_line_candidates,
)

__all__ = [
    'PuzzleClues', 'VoltorbGame', 'FlipRecommendation', 'SolverResult',
    'SearchLimitExceeded', 'InconsistentPuzzleError',
    'heuristic_scores', 'legacy_scores', 'valid_line_candidates', 'solve',
    'SolverMethod', 'Strategy',
]
