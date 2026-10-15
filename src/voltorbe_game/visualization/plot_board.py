"""Voltorb Flip board, probability, and recommendation visualizations.

The module builds on the same ``PlotParameters``, ``create_figure`` and
``plot_image`` helpers used by the Gantt-chart plotting package.
"""

from dataclasses import replace

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ..puzzle import VoltorbGame
from ..solver import SolverResult
from .plot_2d import plot_image
from .plot_basic import configure_axes, create_figure
from .plot_parameters import PlotParameters


def _axis_style(parameters: PlotParameters | None, title: str) -> PlotParameters:
    """Create independent plot settings without mutating the supplied style."""
    return replace(
        parameters or PlotParameters(), title=title, xlabel='Column', ylabel='Row',
        show_legend=False, show_grid=False, aspect='equal',
    )


def _format_board_axes(ax: Axes, game: VoltorbGame) -> None:
    """Show 1-based row and column labels and tile boundaries."""
    n_rows, n_cols = game.clues.rows, game.clues.columns
    ax.set_xticks(range(n_cols), labels=[str(c + 1) for c in range(n_cols)])
    ax.set_yticks(range(n_rows), labels=[str(r + 1) for r in range(n_rows)])
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)


def _mark_recommendation(ax: Axes, result: SolverResult | None) -> None:
    if result is None or result.recommendation is None:
        return
    r, c = result.recommendation.row, result.recommendation.column
    ax.add_patch(Rectangle(
        (c - 0.48, r - 0.48), 0.96, 0.96, fill=False,
        edgecolor='crimson', linewidth=2.7, zorder=8,
    ))


def plot_board(
    game: VoltorbGame,
    *,
    result: SolverResult | None = None,
    parameters: PlotParameters | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Display known tiles and row/column clue totals.

    Parameters
    ----------
    game : VoltorbGame
        Current board and clues.
    result : SolverResult, optional
        Mark the recommended cell with a red outline.
    parameters : PlotParameters, optional
        Shared Matplotlib styling configuration.
    ax : matplotlib.axes.Axes, optional
        Target axes. A figure is created when omitted.

    Returns
    -------
    tuple of matplotlib.figure.Figure and matplotlib.axes.Axes
        Figure and its populated axes.
    """
    opts = _axis_style(parameters, 'Revealed tiles')
    if ax is None:
        figure, ax = create_figure(parameters=opts)
    else:
        figure = ax.figure
    values = game.revealed
    plot_image(ax, values, parameters=replace(opts, colormap='viridis'),
               interpolation='nearest', vmin=-1, vmax=game.clues.max_value)
    for (row, column), value in np.ndenumerate(values):
        ax.text(column, row, '?' if value == -1 else ('V' if value == 0 else str(value)),
                ha='center', va='center', fontsize=opts.fontsize,
                color='black' if value == -1 else 'white', fontweight='bold')
    configure_axes(ax, opts)
    _format_board_axes(ax, game)
    _mark_recommendation(ax, result)
    # Clues beyond plot edges, in row / column order. Rows are shown at right.
    n_cols, n_rows = game.clues.columns, game.clues.rows
    for row, (points, bombs) in enumerate(zip(game.clues.row_points, game.clues.row_voltorbs, strict=True)):
        ax.text(n_cols - 0.30, row, f'{points}/{bombs}', ha='left', va='center', fontsize=opts.tick_size)
    ax.set_xticks(
        range(n_cols),
        labels=[f'{col + 1}\n{points}/{bombs}' for col, (points, bombs) in enumerate(
            zip(game.clues.column_points, game.clues.column_voltorbs, strict=True)
        )],
    )
    ax.set_xlabel('Column', labelpad=8)
    return figure, ax


def plot_analysis(
    game: VoltorbGame,
    result: SolverResult,
    *,
    parameters: PlotParameters | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot known tiles, Voltorb risk and multiplier opportunity side-by-side.

    Exact mode displays true constraint-based probabilities assuming equally
    likely valid boards. Heuristic mode explicitly labels its risk estimates.

    Parameters
    ----------
    game : VoltorbGame
        Current puzzle state.
    result : SolverResult
        Latest analysis (must correspond to current game state).
    parameters : PlotParameters, optional
        Shared figure style.

    Returns
    -------
    tuple
        Matplotlib figure and one-dimensional array of three axes.
    """
    base = replace(parameters or PlotParameters(), figsize=(15, 4.6))
    fig, axes = create_figure(1, 3, parameters=base)
    plot_board(game, result=result, parameters=base, ax=axes[0])

    risk = result.voltorb_risk
    legacy = result.method == 'legacy'
    opportunity = (
        result.scores if legacy else
        result.multiplier_chance if result.multiplier_chance is not None else
        result.scores
    )
    titles = (
        'Voltorb probability' if result.mode == 'exact' else 'Voltorb risk estimate (heuristic)',
        'Original row/column score (∞ = safe)' if legacy else
        'P(2x or 3x)' if result.mode == 'exact' else
        'Multiplier potential (heuristic)',
    )
    for index, (ax, matrix, title) in enumerate(zip(
        axes[1:], (risk, opportunity), titles, strict=True,
    )):
        style = _axis_style(base, title)
        probability_values = result.mode == 'exact' and (index == 0 or not legacy)
        # Imshow masks non-finite numbers; replace legacy's infinite safe scores
        # with a finite high display value, while retaining the ∞ cell labels.
        if legacy and index == 1:
            finite = matrix[np.isfinite(matrix)]
            top = float(finite.max()) if finite.size else 1.0
            display_matrix = np.where(np.isposinf(matrix), max(top, 1.0) + 1.0, matrix)
        else:
            display_matrix = matrix
        plot_image(
            ax, display_matrix, parameters=style,
            vmin=0, vmax=1 if probability_values else None,
        )
        for (row, col), value in np.ndenumerate(matrix):
            if np.isposinf(value):
                label = '∞'
            elif np.isfinite(value):
                label = f'{value:.0%}' if probability_values else f'{value:.2f}'
            else:
                continue
            ax.text(col, row, label, ha='center', va='center', fontsize=style.tick_size - 1)
        configure_axes(ax, style)
        _format_board_axes(ax, game)
        _mark_recommendation(ax, result)
    return fig, axes


__all__ = ['plot_board', 'plot_analysis']
