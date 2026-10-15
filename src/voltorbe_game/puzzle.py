"""Voltorb Flip puzzle clues and mutable revealed-tile state.

The game uses the values 0 (Voltorb), 1, 2, and 3. Unrevealed
cells are represented by -1. Coordinates are zero-based (row, column).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray


def _integers(values: Sequence[int], label: str) -> tuple[int, ...]:
    """Convert a sequence to integers without silently truncating floats."""
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError(f"'{label}' must be a sequence of integers.") from exc
    if not result or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in result):
        raise TypeError(f"'{label}' must be a non-empty sequence of integers.")
    return tuple(int(v) for v in result)


@dataclass(frozen=True, slots=True, kw_only=True)
class PuzzleClues:
    """Immutable point sums and Voltorb counts for each row and column.

    Parameters
    ----------
    row_points, column_points : sequence of int
        Sum of *nonzero* tile values in each row or column.
    row_voltorbs, column_voltorbs : sequence of int
        Number of Voltorb tiles (value zero) in each row or column.
    max_value : int, default 3
        Largest possible tile value. HeartGold/SoulSilver uses 3.

    Raises
    ------
    ValueError
        If dimensions, totals or per-line bounds are inconsistent.
    TypeError
        If clue values are not integers.
    """

    row_points: Sequence[int]
    row_voltorbs: Sequence[int]
    column_points: Sequence[int]
    column_voltorbs: Sequence[int]
    max_value: int = 3

    def __post_init__(self) -> None:
        for name in ('row_points', 'row_voltorbs', 'column_points', 'column_voltorbs'):
            object.__setattr__(self, name, _integers(getattr(self, name), name))
        if isinstance(self.max_value, bool) or not isinstance(self.max_value, Integral) or self.max_value < 1:
            raise ValueError("'max_value' must be a positive integer.")
        object.__setattr__(self, 'max_value', int(self.max_value))
        if len(self.row_points) != len(self.row_voltorbs):
            raise ValueError('Row point and Voltorb clue lengths must agree.')
        if len(self.column_points) != len(self.column_voltorbs):
            raise ValueError('Column point and Voltorb clue lengths must agree.')
        if sum(self.row_points) != sum(self.column_points):
            raise ValueError('Row and column point totals must agree.')
        if sum(self.row_voltorbs) != sum(self.column_voltorbs):
            raise ValueError('Row and column Voltorb totals must agree.')
        for label, sums, bombs, length in (
            ('row', self.row_points, self.row_voltorbs, self.columns),
            ('column', self.column_points, self.column_voltorbs, self.rows),
        ):
            for i, (points, voltorbs) in enumerate(zip(sums, bombs, strict=True)):
                safe = length - voltorbs
                if not 0 <= voltorbs <= length or not safe <= points <= safe * self.max_value:
                    raise ValueError(f'Impossible {label} clue at index {i}: {points} points, {voltorbs} Voltorbs.')

    @property
    def rows(self) -> int:
        """Number of puzzle rows."""
        return len(self.row_points)

    @property
    def columns(self) -> int:
        """Number of puzzle columns."""
        return len(self.column_points)


class VoltorbGame:
    """Store a puzzle and its revealed tiles without doing any plotting.

    Parameters
    ----------
    clues : PuzzleClues
        Validated row and column sums and Voltorb counts.
    revealed : nested sequence or numpy.ndarray, optional
        Initial board with -1 for unknown cells and 0..max_value for
        revealed cells. The data is copied.

    Notes
    -----
    Instances contain no Matplotlib objects or interactive input functions.
    Calling :meth:`reveal` invalidates no external state: solver results are
    immutable snapshots and must be recomputed after a reveal.
    """

    def __init__(
        self,
        clues: PuzzleClues,
        revealed: Sequence[Sequence[int]] | NDArray[np.integer] | None = None,
    ) -> None:
        if not isinstance(clues, PuzzleClues):
            raise TypeError("'clues' must be a PuzzleClues instance.")
        self.clues = clues
        self._revealed: NDArray[np.int64] = np.full((clues.rows, clues.columns), -1, dtype=np.int64)
        if revealed is not None:
            self.set_revealed(revealed)

    @property
    def revealed(self) -> NDArray[np.int64]:
        """Return a copy of the revealed-cell matrix (-1 means hidden)."""
        return self._revealed.copy()

    @property
    def is_lost(self) -> bool:
        """Whether a Voltorb has been revealed."""
        return bool(np.any(self._revealed == 0))

    @property
    def is_won(self) -> bool:
        """Whether all tiles with multipliers greater than 1 are revealed.

        The row clues determine the total sum of positive excess over 1;
        subtracting the excess of revealed tiles gives the remaining amount.
        """
        if self.is_lost:
            return False
        required_excess = sum(self.clues.row_points) - (
            self.clues.rows * self.clues.columns - sum(self.clues.row_voltorbs)
        )
        revealed_excess = int(np.maximum(self._revealed - 1, 0).sum())
        return required_excess == revealed_excess

    def set_revealed(self, board: Sequence[Sequence[int]] | NDArray[np.integer]) -> None:
        """Replace the revealed tiles after validating all known clues.

        Parameters
        ----------
        board : rectangular integer matrix
            Shape must match the puzzle. Values are -1 or 0..max_value.
        """
        arr = np.asarray(board)
        if arr.shape != (self.clues.rows, self.clues.columns):
            raise ValueError(f"Expected a board of shape {(self.clues.rows, self.clues.columns)}.")
        if arr.dtype.kind not in 'iu' or (arr < -1).any() or (arr > self.clues.max_value).any():
            raise ValueError('Board entries must be integers from -1 to max_value.')
        board_int = arr.astype(np.int64, copy=True)
        self._validate_known(board_int)
        self._revealed = board_int

    def reveal(self, row: int, column: int, value: int) -> None:
        """Reveal a cell, or confirm a previously recorded value.

        Raises
        ------
        IndexError
            If coordinates are outside the board.
        ValueError
            If the value is invalid or contradicts the known clues.
        """
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (row, column)):
            raise TypeError('Row and column must be integers.')
        if not (0 <= row < self.clues.rows and 0 <= column < self.clues.columns):
            raise IndexError('Tile position is out of bounds.')
        if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= self.clues.max_value:
            raise ValueError(f"'value' must be an integer between 0 and {self.clues.max_value}.")
        existing = int(self._revealed[row, column])
        if existing >= 0 and existing != value:
            raise ValueError(f'Tile ({row}, {column}) was already revealed as {existing}.')
        proposed = self._revealed.copy()
        proposed[row, column] = value
        self._validate_known(proposed)
        self._revealed = proposed

    def _validate_known(self, board: NDArray[np.int64]) -> None:
        for axis, sums, bombs in (
            (1, self.clues.row_points, self.clues.row_voltorbs),
            (0, self.clues.column_points, self.clues.column_voltorbs),
        ):
            unknown = (board < 0).sum(axis=axis)
            seen_bombs = (board == 0).sum(axis=axis)
            seen_points = np.maximum(board, 0).sum(axis=axis)
            for i, (target, target_bombs) in enumerate(zip(sums, bombs, strict=True)):
                remaining_bombs = target_bombs - int(seen_bombs[i])
                remaining_unknown = int(unknown[i])
                remaining_safe = remaining_unknown - remaining_bombs
                remaining_points = target - int(seen_points[i])
                if (remaining_bombs < 0 or remaining_safe < 0 or
                    not remaining_safe <= remaining_points <= remaining_safe * self.clues.max_value):
                    label = 'row' if axis == 1 else 'column'
                    raise ValueError(f'Revealed tiles contradict {label} clue {i}.')


__all__ = ['PuzzleClues', 'VoltorbGame']
