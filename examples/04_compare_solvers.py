"""Compare the three methods using the original HeartGold/SoulSilver puzzle.

Execute from the repository root after installing the project:
    python examples/04_compare_solvers.py
"""

from voltorbe_game import solve
from voltorbe_game.simulation import example_game


def main() -> None:
    game = example_game()
    for method in ('exact', 'legacy', 'heuristic'):
        result = solve(game, method=method)
        print(f'[{result.method}] analysis={result.mode}, valid boards={result.possible_grids}')
        if result.recommendation is None:
            print('  No useful hidden tile recommended.')
            continue
        suggestion = result.recommendation
        print(f'  Flip row {suggestion.row + 1}, column {suggestion.column + 1}')
        label = 'Probability' if result.mode == 'exact' else 'Risk estimate'
        print(f'  {label} of Voltorb: {suggestion.voltorb_risk:.1%}')
        if suggestion.multiplier_chance is not None:
            print(f'  P(2x or 3x): {suggestion.multiplier_chance:.1%}')


if __name__ == '__main__':
    main()
