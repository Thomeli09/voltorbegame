"""Recompute and visualize probabilities after a known safe reveal."""

from pathlib import Path

from voltorbe_game import solve
from voltorbe_game.simulation import example_game
from voltorbe_game.visualization import PlotParameters, plot_analysis, save_figure

output = Path('voltorb_analysis.png')
game = example_game()

# Column 5 has no Voltorbs. Its row-2 tile has a substantial
# (but not certain) chance of containing a 2x or 3x multiplier.
result = solve(game)
print('Recommended tile:', result.recommendation)

style = PlotParameters(figsize=(15, 5), fontsize=11)
fig, axes = plot_analysis(game, result, parameters=style)
save_figure(fig, output)
print(f'Figure saved to {output.resolve()}')

# Example of updating your knowledge after looking at a real tile.
# Only pass a value you actually observed during play.
# game.reveal(row=1, column=4, value=2)
# next_result = solve(game)
