"""Matplotlib-based, typed 2D plotting utilities for the Voltorb Flip package.

The public visualization API is deliberately centered on explicit figures,
axes, plot configuration and returned Matplotlib artists.
"""

from .plot_basic import (
    add_colorbar,
    add_figure_text,
    add_legend,
    add_legend_with_subtitles,
    close_all_figures,
    close_figure,
    compute_limits,
    configure_axes,
    create_figure,
    format_text,
    maximize_figure,
    save_figure,
    set_figure_size,
    set_latex_text,
    show_figure,
    update_layout,
    update_line_colors,
)
from .plot_board import plot_analysis, plot_board
from .plot_parameters import PlotParameters
from .plot_2d import (
    draw_circle,
    draw_infinite_line,
    draw_reference_line,
    draw_span,
    fill_area,
    plot_bars,
    plot_empty_line,
    plot_histogram,
    plot_histogram2d,
    plot_image,
    plot_line,
    plot_pie,
    plot_series,
    plot_text,
    update_line,
)

__all__ = [
    "PlotParameters", "plot_board", "plot_analysis",
    "create_figure", "configure_axes", "add_legend", "add_legend_with_subtitles",
    "update_line_colors", "add_colorbar", "format_text", "add_figure_text",
    "compute_limits", "set_figure_size", "save_figure", "show_figure",
    "close_figure", "close_all_figures", "update_layout", "set_latex_text",
    "maximize_figure", "plot_text", "plot_line", "plot_empty_line",
    "update_line", "plot_series", "draw_reference_line", "draw_infinite_line",
    "draw_span", "fill_area", "plot_bars", "plot_histogram",
    "plot_histogram2d", "plot_pie", "plot_image", "draw_circle",
]
