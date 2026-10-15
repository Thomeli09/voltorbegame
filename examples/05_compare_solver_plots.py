"""Export analysis images from the exact and legacy solvers.

Both methods share exactly the same puzzle data and visualization utilities;
the legacy plot displays the original row/column score in its third panel.
"""

from pathlib import Path

from voltorbe_game import solve
from voltorbe_game.simulation import example_game
from voltorbe_game.visualization import PlotParameters, plot_analysis, save_figure


def main() -> None:
    game = example_game()
    style = PlotParameters(fontsize=11)
    for method in ('exact', 'legacy'):
        result = solve(game, method=method)
        fig, _ = plot_analysis(game, result, parameters=style)
        filename = Path(f'voltorb_{method}.png')
        save_figure(fig, filename)
        print(f'Saved {filename}')
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == '__main__':
    main()
