"""Demonstrate exact small-puzzle probabilities and labeled fallback."""

from voltorbe_game import PuzzleClues, VoltorbGame, solve

clues = PuzzleClues(
    row_points=[2, 2], row_voltorbs=[1, 1],
    column_points=[2, 2], column_voltorbs=[1, 1],
)
game = VoltorbGame(clues)

exact = solve(game)
print('Exactly matching boards:', exact.possible_grids)
print('Exact Voltorb probabilities:\n', exact.voltorb_risk)

game.reveal(0, 0, 2)
updated = solve(game)
print('Remaining boards:', updated.possible_grids)
print('Recommended next tile:', updated.recommendation)

# This option is for large searches only: it does NOT provide calibrated
# probabilities or a count of possible boards.
fallback = solve(game, max_nodes=1, fallback=True)
print('Fallback analysis:', fallback.mode)
