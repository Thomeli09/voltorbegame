"""Command-line example and interactive Voltorb Flip solving session.

Use ``python -m voltorbe_game.simulation`` after installing the package.
All input()/print() calls live here, rather than in the puzzle or solver.
"""

import argparse
from collections.abc import Sequence

from .. import PuzzleClues, SearchLimitExceeded, VoltorbGame, solve
from ..visualization import plot_analysis


def example_game() -> VoltorbGame:
    """Return the same 5x5 clue set used in the original VoltorbeGame.py."""
    return VoltorbGame(PuzzleClues(
        column_points=[5, 7, 3, 5, 8],
        column_voltorbs=[2, 1, 3, 1, 0],
        row_points=[5, 8, 5, 5, 5],
        row_voltorbs=[2, 1, 0, 2, 2],
    ))


def main(argv: Sequence[str] | None = None) -> int:
    """Solve the original example, optionally flipping tiles interactively."""
    parser = argparse.ArgumentParser(description='Voltorb Flip solver example')
    parser.add_argument('--once', action='store_true', help='Show a single recommendation without prompting')
    parser.add_argument('--plot', action='store_true', help='Show analysis charts at each iteration')
    parser.add_argument('--fallback', action='store_true', help='Allow labeled heuristics if exact search is too large')
    parser.add_argument('--max-nodes', type=int, default=2_000_000)
    parser.add_argument('--method', choices=['exact', 'legacy', 'heuristic'], default='exact',
                        help='Select the solver engine (default: exact)')
    parser.add_argument('--strategy', choices=['risk_first', 'expected_gain', 'legacy_score'],
                        default=None, help='Selection policy (defaults to the method-specific policy)')
    args = parser.parse_args(argv)
    game = example_game()
    while True:
        try:
            result = solve(
                game, method=args.method, strategy=args.strategy,
                max_nodes=args.max_nodes, fallback=args.fallback,
            )
        except (SearchLimitExceeded, ValueError) as exc:
            parser.error(str(exc))
        print(f'Solver: {result.method} ({result.strategy}); analysis: {result.mode}.')
        boards = result.possible_grids if result.possible_grids is not None else 'not computed'
        print(f'Valid boards: {boards}.')
        if game.is_won:
            print('Puzzle solved: every 2x/3x tile has been revealed.')
            return 0
        if game.is_lost:
            print('A Voltorb was revealed; this game is lost.')
            return 1
        rec = result.recommendation
        if rec is None:
            print('No useful recommendation under the current analysis.')
            return 2
        print(f'Suggested flip: row {rec.row + 1}, column {rec.column + 1}.')
        print(f'  {"Exact Voltorb probability" if result.mode == "exact" else "Uncalibrated risk estimate"}: '
              f'{rec.voltorb_risk:.1%}')
        if rec.multiplier_chance is not None:
            print(f'  P(2x or 3x): {rec.multiplier_chance:.1%}')
        if args.plot:
            import matplotlib.pyplot as plt
            fig, _ = plot_analysis(game, result)
            plt.show(block=False)
            plt.pause(0.05)
        if args.once:
            return 0
        reply = input('Revealed value (0, 1, 2, 3), or q to quit: ').strip().lower()
        if reply in ('q', 'quit', 'exit'):
            return 0
        try:
            game.reveal(rec.row, rec.column, int(reply))
        except (ValueError, TypeError) as exc:
            print(f'Invalid input: {exc}')
            continue


__all__ = ['example_game', 'main']
