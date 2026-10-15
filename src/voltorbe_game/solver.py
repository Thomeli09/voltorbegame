"""Constraint-based Voltorb Flip analysis and flip recommendations.

Exact mode counts every board satisfying the clues and already revealed
cells. It assumes these boards are equally likely; the real game may use
additional level-specific board-generation restrictions.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from math import prod
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .puzzle import VoltorbGame
from .solvers.legacy import choose_legacy_tile, legacy_scores


type Strategy = Literal['risk_first', 'expected_gain', 'legacy_score']
type SolverMethod = Literal['exact', 'legacy', 'heuristic']
type AnalysisMode = Literal['exact', 'heuristic']


class SearchLimitExceeded(RuntimeError):
    """Raised when exact enumeration exceeds its search-node budget."""


class InconsistentPuzzleError(ValueError):
    """Raised when no board can satisfy all the clues and revealed cells."""


@dataclass(frozen=True, slots=True)
class FlipRecommendation:
    """Suggested hidden tile, addressed with zero-based row and column.

    Attributes
    ----------
    row, column : int
        Position of the tile.
    voltorb_risk : float
        Exact Voltorb probability or a labeled heuristic estimate, 0..1.
    multiplier_chance : float, optional
        Exact probability that the tile is 2 or 3; ``None`` in heuristic mode.
    strategy : {'risk_first', 'expected_gain', 'legacy_score'}
        The ranking rule used to select this tile.
    """

    row: int
    column: int
    voltorb_risk: float
    multiplier_chance: float | None
    strategy: Strategy


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Snapshot of one analysis of a :class:`~voltorbe_game.VoltorbGame`.

    ``probabilities[row, column, value]`` is available only in exact mode.
    Probabilities represent fractions, not percentages. Unknown cells in
    exact mode have distributions summing to one; known cells have a unit
    probability on their known value. Heuristic risk estimates are *not*
    probabilities and are not calibrated.
    """

    mode: AnalysisMode
    strategy: Strategy
    possible_grids: int | None
    probabilities: NDArray[np.float64] | None
    voltorb_risk: NDArray[np.float64]
    multiplier_chance: NDArray[np.float64] | None
    scores: NDArray[np.float64]
    recommendation: FlipRecommendation | None
    visited_nodes: int
    elapsed_seconds: float
    method: SolverMethod = 'exact'


def valid_line_candidates(
    points: int, voltorbs: int, known: Sequence[int], *, max_value: int = 3,
) -> list[tuple[int, ...]]:
    """Generate all line assignments satisfying its points and Voltorbs.

    Parameters
    ----------
    points, voltorbs : int
        Sum of tile values and number of zero-valued cells in the line.
    known : sequence of int
        Revealed values; -1 represents a hidden cell.
    max_value : int, default 3
        Largest permitted tile value.

    Returns
    -------
    list of tuple of int
        Valid lines in deterministic lexicographic order.
    """
    # Fast enough for ordinary 5x5 Voltorb Flip puzzles.
    options = [range(max_value + 1) if value < 0 else (int(value),) for value in known]
    return [
        candidate for candidate in product(*options)
        if sum(candidate) == points and candidate.count(0) == voltorbs
    ]


def _enumerate_counts(
    game: VoltorbGame, max_nodes: int,
) -> tuple[NDArray[np.int64], int, int]:
    """Enumerate exact solutions with column-prefix constraint pruning.

    Orient the recursion to minimize the product of candidate line counts.
    During recursion, every column prefix must belong to at least one fully
    valid column candidate. We accumulate value counts rather than retaining
    every complete grid, so memory depends mainly on line-candidate counts.
    """
    clues = game.clues
    known = game.revealed
    row_options = [valid_line_candidates(p, b, known[i], max_value=clues.max_value)
                   for i, (p, b) in enumerate(zip(clues.row_points, clues.row_voltorbs, strict=True))]
    col_options = [valid_line_candidates(p, b, known[:, j], max_value=clues.max_value)
                   for j, (p, b) in enumerate(zip(clues.column_points, clues.column_voltorbs, strict=True))]
    if any(not options for options in row_options + col_options):
        raise InconsistentPuzzleError('No line assignments satisfy the given clues.')

    transpose = prod(map(len, col_options)) < prod(map(len, row_options))
    if transpose:
        lines, cross_lines = col_options, row_options
        rows, columns = clues.columns, clues.rows
    else:
        lines, cross_lines = row_options, col_options
        rows, columns = clues.rows, clues.columns

    prefix_sets: list[list[set[tuple[int, ...]]]] = []
    for candidates in cross_lines:
        prefix_sets.append([
            {candidate[:depth] for candidate in candidates}
            for depth in range(rows + 1)
        ])

    value_counts = np.zeros((rows, columns, clues.max_value + 1), dtype=np.int64)
    working = np.empty((rows, columns), dtype=np.int64)
    prefixes: list[tuple[int, ...]] = [()] * columns
    visits = 0
    solution_count = 0

    def search(depth: int) -> None:
        nonlocal visits, solution_count
        if depth == rows:
            solution_count += 1
            # Each completed grid contributes one observation for each tile.
            for row_index in range(rows):
                for column_index in range(columns):
                    value_counts[row_index, column_index, working[row_index, column_index]] += 1
            return
        for candidate in lines[depth]:
            visits += 1
            if visits > max_nodes:
                raise SearchLimitExceeded(
                    f'Exact search exceeded {max_nodes:,} candidate nodes; '
                    'increase max_nodes or enable heuristic fallback.'
                )
            next_prefixes = [prefixes[column] + (candidate[column],)
                             for column in range(columns)]
            if any(next_prefixes[column] not in prefix_sets[column][depth + 1]
                   for column in range(columns)):
                continue
            previous = prefixes.copy()
            prefixes[:] = next_prefixes
            working[depth, :] = candidate
            search(depth + 1)
            prefixes[:] = previous

    search(0)
    if not solution_count:
        raise InconsistentPuzzleError('No complete grid satisfies all row and column clues.')
    if transpose:
        value_counts = value_counts.transpose(1, 0, 2)
    return value_counts, solution_count, visits


def heuristic_scores(game: VoltorbGame) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate risk and potential without enumerating complete boards.

    The risk combines remaining Voltorb fractions for the intersecting row
    and column. These are only *heuristics*, not posterior probabilities.
    A cell with zero possible multiplier benefit gets score zero.

    Returns
    -------
    tuple of numpy.ndarray
        Risk estimates in [0, 1] and non-negative potential scores.
    """
    clues, known = game.clues, game.revealed
    rows, cols = known.shape
    risk = np.ones((rows, cols), dtype=float)
    scores = np.zeros((rows, cols), dtype=float)

    def line_factors(values: NDArray[np.int64], points: int, bombs: int) -> tuple[float, float]:
        unknown = int(np.count_nonzero(values < 0))
        if unknown == 0:
            return 1.0, 0.0
        unseen_bombs = bombs - int(np.count_nonzero(values == 0))
        unseen_safe = unknown - unseen_bombs
        unseen_points = points - int(np.maximum(values, 0).sum())
        excess = max(0, unseen_points - unseen_safe)
        risk_factor = unseen_bombs / unknown
        benefit = excess / unseen_safe if unseen_safe else 0.0
        return risk_factor, benefit

    row_factors = [line_factors(known[r], clues.row_points[r], clues.row_voltorbs[r])
                   for r in range(rows)]
    col_factors = [line_factors(known[:, c], clues.column_points[c], clues.column_voltorbs[c])
                   for c in range(cols)]
    for r in range(rows):
        for c in range(cols):
            if known[r, c] >= 0:
                risk[r, c] = np.nan
                continue
            risk[r, c] = min(row_factors[r][0], col_factors[c][0])
            scores[r, c] = (1.0 - risk[r, c]) * min(row_factors[r][1], col_factors[c][1])
    return risk, scores


def _recommend(
    game: VoltorbGame,
    risk: NDArray[np.float64],
    scores: NDArray[np.float64],
    multiplier_chance: NDArray[np.float64] | None,
    strategy: Strategy,
) -> FlipRecommendation | None:
    known = game.revealed
    if game.is_won or game.is_lost:
        return None
    choices: list[tuple[tuple[float, ...], int, int]] = []
    for r, c in zip(*np.where(known < 0), strict=True):
        if multiplier_chance is not None:
            opportunity = float(multiplier_chance[r, c])
        else:
            opportunity = float(scores[r, c])
        if opportunity <= 1e-12:
            continue  # Cannot help find an unrevealed 2x/3x (or no heuristic evidence).
        danger = float(risk[r, c])
        if strategy == 'risk_first':
            key = (danger, -opportunity, -float(scores[r, c]), int(r), int(c))
        else:
            reward = (1 - danger) * opportunity
            key = (-reward, danger, -opportunity, int(r), int(c))
        choices.append((key, int(r), int(c)))
    if not choices:
        return None
    _, row, column = min(choices)
    return FlipRecommendation(
        row=row, column=column, voltorb_risk=float(risk[row, column]),
        multiplier_chance=(float(multiplier_chance[row, column])
                           if multiplier_chance is not None else None),
        strategy=strategy,
    )


def _heuristic_result(
    game: VoltorbGame,
    *,
    method: SolverMethod,
    strategy: Strategy,
    started: float,
    visited_nodes: int = 0,
) -> SolverResult:
    """Build an explicitly approximate result without enumerating boards."""
    risk, potential = heuristic_scores(game)
    if method == 'legacy':
        scores = legacy_scores(game)
        location = choose_legacy_tile(game, risk, scores)
        recommendation = (
            FlipRecommendation(
                row=location[0], column=location[1],
                voltorb_risk=float(risk[location]),
                multiplier_chance=None, strategy='legacy_score',
            )
            if location is not None else None
        )
    else:
        scores = potential
        recommendation = _recommend(game, risk, scores, None, strategy)
    for array in (risk, scores):
        array.setflags(write=False)
    return SolverResult(
        mode='heuristic', strategy=strategy, possible_grids=None,
        probabilities=None, voltorb_risk=risk, multiplier_chance=None,
        scores=scores, recommendation=recommendation,
        visited_nodes=visited_nodes, elapsed_seconds=perf_counter() - started,
        method=method,
    )


def solve(
    game: VoltorbGame,
    *,
    method: SolverMethod = 'exact',
    strategy: Strategy | None = None,
    max_nodes: int = 2_000_000,
    fallback: bool = False,
) -> SolverResult:
    """Analyze a puzzle using a selectable solver method.

    Parameters
    ----------
    game : VoltorbGame
        Validated clues and revealed tiles.
    method : {'exact', 'legacy', 'heuristic'}, default 'exact'
        ``'exact'`` enumerates all boards subject to the constraints and
        ranks tiles with a risk-first or expected-gain rule. ``'legacy'``
        reuses those exact board frequencies but applies the row/column
        scoring and risk-based selection from the original implementation.
        ``'heuristic'`` skips full enumeration and uses fast approximate
        remaining-clue estimates.
    strategy : {'risk_first', 'expected_gain', 'legacy_score'}, optional
        Selection rule. Defaults to ``'legacy_score'`` for ``method='legacy'``
        and ``'risk_first'`` otherwise. ``'legacy_score'`` is valid only
        with ``method='legacy'``; other strategies cannot be combined with
        that method.
    max_nodes : int, default 2000000
        Maximum branches for exact enumeration. Ignored for direct heuristic
        evaluation (but must still be positive).
    fallback : bool, default False
        If exact enumeration exceeds the node budget, return a result
        explicitly labeled ``mode='heuristic'`` instead of raising.
        The fallback retains the selected method's scoring policy.

    Returns
    -------
    SolverResult
        Snapshot of the chosen solver method, probabilities if enumerated,
        scores, and recommended hidden tile. Probabilities are fractions.

    Raises
    ------
    ValueError
        If method, strategy or maximum node count is invalid.
    InconsistentPuzzleError
        If enumeration proves there are no valid boards.
    SearchLimitExceeded
        If enumeration exceeds the budget and fallback is disabled.

    Notes
    -----
    Enumeration assumes all constraint-compatible boards are equally likely.
    This is not necessarily the same as the actual game's board distribution.
    ``'legacy'`` recreates the old *ranking approach* using the new memory-
    efficient enumeration engine; it does not retain the old interactive
    input loop or store every possible complete board.
    """
    if method not in ('exact', 'legacy', 'heuristic'):
        raise ValueError("method must be 'exact', 'legacy', or 'heuristic'.")
    if strategy is None:
        strategy = 'legacy_score' if method == 'legacy' else 'risk_first'
    if method == 'legacy':
        if strategy != 'legacy_score':
            raise ValueError("method='legacy' requires strategy='legacy_score'.")
    elif strategy not in ('risk_first', 'expected_gain'):
        raise ValueError(f"strategy={strategy!r} cannot be used with method={method!r}.")
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
        raise ValueError('max_nodes must be a positive integer.')
    start_time = perf_counter()

    if method == 'heuristic':
        return _heuristic_result(
            game, method=method, strategy=strategy, started=start_time,
        )

    try:
        counts, number, visited = _enumerate_counts(game, max_nodes)
    except SearchLimitExceeded:
        if not fallback:
            raise
        return _heuristic_result(
            game, method=method, strategy=strategy, started=start_time,
            visited_nodes=max_nodes,
        )

    probabilities = counts.astype(float) / number
    risk = probabilities[:, :, 0].copy()
    chance = probabilities[:, :, 2:].sum(axis=-1)
    known = game.revealed

    if method == 'legacy':
        scores = legacy_scores(game)
        # Previously revealed tiles remain excluded from all recommendations.
        risk[known >= 0] = np.nan
        chance[known >= 0] = 0
        location = choose_legacy_tile(game, risk, scores, chance)
        recommendation = (
            FlipRecommendation(
                row=location[0], column=location[1],
                voltorb_risk=float(risk[location]),
                multiplier_chance=float(chance[location]),
                strategy='legacy_score',
            )
            if location is not None else None
        )
    else:
        # A multiplier contributes its face value; 0/1 are not target tiles.
        reward_weights = np.arange(game.clues.max_value + 1, dtype=float)
        reward_weights[:2] = 0.0
        scores = probabilities @ reward_weights
        risk[known >= 0] = np.nan
        scores[known >= 0] = 0
        chance[known >= 0] = 0
        recommendation = _recommend(game, risk, scores, chance, strategy)

    for array in (probabilities, risk, scores, chance):
        array.setflags(write=False)
    return SolverResult(
        mode='exact', method=method, strategy=strategy, possible_grids=number,
        probabilities=probabilities, voltorb_risk=risk,
        multiplier_chance=chance, scores=scores, recommendation=recommendation,
        visited_nodes=visited, elapsed_seconds=perf_counter() - start_time,
    )


__all__ = [
    'AnalysisMode', 'SolverMethod', 'Strategy', 'FlipRecommendation', 'SolverResult',
    'SearchLimitExceeded', 'InconsistentPuzzleError',
    'valid_line_candidates', 'heuristic_scores', 'legacy_scores', 'solve',
]
