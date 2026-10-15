"""Legacy Voltorb Flip score method, isolated from exact enumeration.

This module preserves the row/column-based score formula from the original
``VoltorbeGameLib.CmptScores`` implementation. The old interactive solver used
these scores to filter candidate tiles and then preferred the lowest Voltorb
frequency among feasible boards. Exact board frequencies are computed by the
separate enumeration engine so we do not retain every complete board.

An infinite score indicates that at least one intersecting line has no
remaining Voltorbs and does not have zero available safe cells. It is not a
probability, and neither the finite scores nor the heuristic risk estimates
are calibrated measures of success.
"""

import numpy as np
from numpy.typing import NDArray

from ..puzzle import VoltorbGame


def legacy_scores(game: VoltorbGame) -> NDArray[np.float64]:
    """Calculate the original row/column score for each hidden tile.

    The finite line score is the number of remaining points divided by the
    number of remaining safe positions. If no unknown safe position remains,
    the line score is zero; if no Voltorbs remain and safe positions exist,
    the line score is infinite. A tile receives the minimum row/column score,
    with scores of one or less set to zero, except that an infinite line
    score makes a hidden tile a forced-safe candidate.

    Parameters
    ----------
    game : VoltorbGame
        Puzzle clues and current revealed tiles.

    Returns
    -------
    numpy.ndarray of float
        Scores with shape ``(rows, columns)``. Revealed cells are set to
        zero. Infinite scores signify cells considered safe by line clues.

    Notes
    -----
    This is a *heuristic score*, not the expected value or the probability
    of finding a multiplier. It intentionally retains the original formula.
    """
    clues = game.clues
    known = game.revealed
    rows, columns = known.shape

    def line_score(line: NDArray[np.int64], points: int, bombs: int) -> float:
        unknown = int(np.count_nonzero(line < 0))
        remaining_bombs = bombs - int(np.count_nonzero(line == 0))
        remaining_safe = unknown - remaining_bombs
        if remaining_safe == 0:
            return 0.0
        if remaining_bombs <= 0:
            return float('inf')
        remaining_points = points - int(np.maximum(line, 0).sum())
        return remaining_points / remaining_safe

    row_scores = [
        line_score(known[r], clues.row_points[r], clues.row_voltorbs[r])
        for r in range(rows)
    ]
    column_scores = [
        line_score(known[:, c], clues.column_points[c], clues.column_voltorbs[c])
        for c in range(columns)
    ]
    scores = np.zeros((rows, columns), dtype=np.float64)
    for row in range(rows):
        for col in range(columns):
            if known[row, col] >= 0:
                continue
            row_score, col_score = row_scores[row], column_scores[col]
            score = min(row_score, col_score)
            if score <= 1:
                score = 0.0
            if np.isinf(row_score) or np.isinf(col_score):
                score = float('inf')
            scores[row, col] = score
    return scores


def choose_legacy_tile(
    game: VoltorbGame,
    risk: NDArray[np.float64],
    scores: NDArray[np.float64],
    multiplier_chance: NDArray[np.float64] | None = None,
) -> tuple[int, int] | None:
    """Choose a tile with the original score-first, minimum-risk policy.

    Forced-safe tiles (infinite score) are preferred. Otherwise, the policy
    minimizes the estimated Voltorb risk among positive-score candidates,
    breaking ties in row-major order, as NumPy's ``argmin`` does. If exact
    frequencies are available, tiles with zero multiplier chance are excluded.

    Parameters
    ----------
    game : VoltorbGame
        Puzzle state.
    risk : numpy.ndarray
        Exact Voltorb probabilities or explicitly heuristic estimates.
    scores : numpy.ndarray
        Output of :func:`legacy_scores`.
    multiplier_chance : numpy.ndarray, optional
        Exact chance of finding a 2x or 3x tile. If available, exclude tiles
        that cannot contribute to winning the game.

    Returns
    -------
    tuple of int or None
        Zero-based ``(row, column)``, or None if no useful tile is known.
    """
    if game.is_won or game.is_lost:
        return None
    unknown = game.revealed < 0
    eligible = unknown & (scores > 1)
    if multiplier_chance is not None:
        eligible &= multiplier_chance > 0
        if not np.any(eligible):
            eligible = unknown & (multiplier_chance > 0)
    if not np.any(eligible):
        return None
    safe = eligible & np.isinf(scores)
    if np.any(safe):
        eligible = safe
    positions = np.argwhere(eligible)
    risks = np.array([risk[row, col] for row, col in positions], dtype=float)
    # argmin is stable: ties keep the first position in row-major order.
    row, col = positions[int(np.argmin(risks))]
    return int(row), int(col)


__all__ = ['legacy_scores', 'choose_legacy_tile']
