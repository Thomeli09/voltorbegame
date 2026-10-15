"""Board and general-purpose plotting integration tests."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from voltorbe_game import solve
from voltorbe_game.simulation import example_game, main
from voltorbe_game.visualization import (
    PlotParameters, create_figure, plot_analysis, plot_board, plot_line, save_figure,
)


def test_board_plot(tmp_path):
    game = example_game()
    figure, ax = plot_board(game, result=solve(game))
    assert ax.get_xlabel() == 'Column'
    assert len(ax.texts) >= 25
    target = tmp_path / 'board.png'
    save_figure(figure, target)
    assert target.exists()
    plt.close(figure)


def test_analysis_three_panels(tmp_path):
    game = example_game()
    fig, axes = plot_analysis(game, solve(game))
    assert len(axes) == 3
    assert 'probability' in axes[1].get_title().lower()
    fig.savefig(tmp_path / 'analysis.png')
    plt.close(fig)


def test_generic_figure_helpers():
    params = PlotParameters(title='General plot', show_grid=True)
    fig, ax = create_figure(parameters=params)
    line = plot_line(ax, [1, 2], [2, 4], parameters=params)
    assert line is not None
    plt.close(fig)


def test_cli_once(capsys):
    assert main(['--once']) == 0
    out = capsys.readouterr().out
    assert 'Suggested flip' in out


def test_legacy_score_panel_is_labeled_and_shows_infinity(tmp_path):
    game = example_game()
    result = solve(game, method='legacy')
    fig, axes = plot_analysis(game, result)
    assert 'row/column score' in axes[2].get_title().lower()
    assert any(text.get_text() == '∞' for text in axes[2].texts)
    fig.savefig(tmp_path / 'legacy.png')
    plt.close(fig)


def test_heuristic_analysis_is_clearly_labeled(tmp_path):
    game = example_game()
    result = solve(game, method='heuristic')
    fig, axes = plot_analysis(game, result)
    assert 'heuristic' in axes[1].get_title().lower()
    fig.savefig(tmp_path / 'heuristic.png')
    plt.close(fig)
