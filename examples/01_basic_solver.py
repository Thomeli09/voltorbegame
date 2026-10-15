"""Basic solver for the original HeartGold/SoulSilver clue set."""

from voltorbe_game import PuzzleClues, VoltorbGame, solve

clues = PuzzleClues(
    column_points=[5, 7, 3, 5, 8],
    column_voltorbs=[2, 1, 3, 1, 0],
    row_points=[5, 8, 5, 5, 5],
    row_voltorbs=[2, 1, 0, 2, 2],
)
game = VoltorbGame(clues)
result = solve(game)
print(f'{result.possible_grids} valid boards')
if result.recommendation is not None:
    rec = result.recommendation
    print(f'Flip row {rec.row + 1}, column {rec.column + 1}')
    print(f'Voltorb risk: {rec.voltorb_risk:.1%}')
    print(f'P(multiplier): {rec.multiplier_chance:.1%}')
