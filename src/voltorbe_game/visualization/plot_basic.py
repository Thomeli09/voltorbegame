"""Figure creation, formatting, legends, and export helpers.

Functions accept explicit Matplotlib figures or axes. They neither silently
create plots nor call ``plt.show`` unless specifically requested.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.text import Text
from numpy.typing import NDArray

from .plot_parameters import ColorValue, PlotParameters


def create_figure(
    rows: int = 1,
    columns: int = 1,
    *,
    parameters: PlotParameters | None = None,
    constrained_layout: bool = True,
    squeeze: bool = True,
) -> tuple[Figure, Axes | NDArray[Any]]:
    """Create a Matplotlib figure and one or more axes.

    Parameters
    ----------
    rows, columns : int, default 1
        Number of subplot rows and columns.
    parameters : PlotParameters, optional
        Configuration supplying the figure dimensions.
    constrained_layout : bool, default True
        Enable Matplotlib's automatic layout engine.
    squeeze : bool, default True
        Return a single Axes for a 1x1 grid; otherwise return an array.

    Returns
    -------
    tuple of matplotlib.figure.Figure and Axes or numpy.ndarray
        Newly created figure and its axes.

    Raises
    ------
    ValueError
        If the requested grid dimensions are not positive.
    """
    if rows < 1 or columns < 1:
        raise ValueError("'rows' and 'columns' must be positive.")
    options = parameters or PlotParameters()
    return plt.subplots(
        rows, columns, figsize=options.figsize,
        layout="constrained" if constrained_layout else None,
        squeeze=squeeze,
    )


def configure_axes(ax: Axes, parameters: PlotParameters) -> Axes:
    """Apply titles, labels, scales, limits, grid, dates, box and legend.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes, containing any artists to be included in its legend.
    parameters : PlotParameters
        Plot appearance settings.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes.
    """
    if parameters.title is not None:
        ax.set_title(parameters.title, fontsize=parameters.title_size)
    if parameters.xlabel is not None:
        ax.set_xlabel(parameters.xlabel, fontsize=parameters.fontsize)
    if parameters.ylabel is not None:
        ax.set_ylabel(parameters.ylabel, fontsize=parameters.fontsize)
    ax.tick_params(axis="both", labelsize=parameters.tick_size)
    ax.set_xscale(parameters.xscale)
    ax.set_yscale(parameters.yscale)
    if parameters.xlim is not None:
        ax.set_xlim(*parameters.xlim)
    if parameters.ylim is not None:
        ax.set_ylim(*parameters.ylim)
    if parameters.aspect is not None:
        ax.set_aspect(parameters.aspect)
    if parameters.show_grid:
        ax.grid(
            True, axis=parameters.grid_axis,
            color=parameters.grid_color or "0.5",
            linestyle=parameters.grid_linestyle,
            linewidth=parameters.grid_linewidth,
            alpha=parameters.grid_alpha,
        )
    else:
        ax.grid(False, axis=parameters.grid_axis)
    for spine in ax.spines.values():
        spine.set_visible(parameters.show_box)
    if parameters.date_format is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(parameters.date_format))
    if parameters.rotate_dates:
        ax.figure.autofmt_xdate()
    if parameters.show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles and any(not label.startswith("_") for label in labels):
            add_legend(ax, parameters)
    return ax


def add_legend(
    ax: Axes,
    parameters: PlotParameters | None = None,
    *,
    title: str | None = None,
    outside: bool | None = None,
    handles: Sequence[Artist] | None = None,
) -> Legend:
    """Create a legend for labeled artists on an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes containing the labeled artists.
    parameters : PlotParameters, optional
        Default legend appearance settings.
    title : str, optional
        Override the configured legend title.
    outside : bool, optional
        Override whether the legend is positioned outside the axes.
    handles : sequence of matplotlib.artist.Artist, optional
        Explicit legend handles for composite plots such as grouped Gantt
        charts. If omitted, the axes artists are detected automatically.

    Returns
    -------
    matplotlib.legend.Legend
        Legend object (may be empty if no labeled artists exist).
    """
    options = parameters or PlotParameters()
    use_outside = options.legend_outside if outside is None else outside
    kwargs: dict[str, Any] = {
        "title": title if title is not None else options.legend_title,
        "fontsize": options.legend_size,
        "title_fontsize": options.legend_size * 1.1,
    }
    if use_outside:
        kwargs.update(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    else:
        kwargs["loc"] = options.legend_location
    if handles is not None:
        kwargs["handles"] = list(handles)
    return ax.legend(**kwargs)


def add_legend_with_subtitles(
    ax: Axes,
    title: str,
    subtitles: Sequence[str],
    positions: Sequence[int],
    *,
    parameters: PlotParameters | None = None,
    subtitle_size_ratio: float = 1.0,
) -> Legend:
    """Insert non-plotted subtitle lines into an existing artist legend.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes with labeled artists.
    title : str
        Legend heading.
    subtitles : sequence of str
        Subheadings to insert in the legend.
    positions : sequence of int
        Zero-based insertion positions in the assembled legend.
    parameters : PlotParameters, optional
        Legend appearance settings.
    subtitle_size_ratio : float, default 1
        Factor applied to the legend font size for the subheadings.

    Returns
    -------
    matplotlib.legend.Legend
        Created legend.

    Raises
    ------
    ValueError
        If subtitles and positions have different lengths.
    """
    if len(subtitles) != len(positions):
        raise ValueError("'subtitles' and 'positions' must have equal lengths.")
    options = parameters or PlotParameters()
    handles, labels = ax.get_legend_handles_labels()
    subtitle_handles: list[Line2D] = []
    for subtitle, position in sorted(zip(subtitles, positions, strict=True), key=lambda pair: pair[1]):
        if not 0 <= position <= len(handles):
            raise ValueError(f"Invalid legend insertion position: {position}.")
        subtitle_handle = Line2D([], [], color="none", linestyle="none")
        handles.insert(position, subtitle_handle)
        labels.insert(position, subtitle)
        subtitle_handles.append(subtitle_handle)
    kwargs: dict[str, Any] = {
        "handles": handles, "labels": labels, "title": title,
        "fontsize": options.legend_size,
        "title_fontsize": options.legend_size * 1.1,
    }
    if options.legend_outside:
        kwargs.update(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    else:
        kwargs["loc"] = options.legend_location
    legend = ax.legend(**kwargs)
    for subtitle_handle in subtitle_handles:
        label = legend.get_texts()[handles.index(subtitle_handle)]
        label.set_fontweight("bold")
        label.set_fontsize(options.legend_size * subtitle_size_ratio)
    return legend


def update_line_colors(
    ax: Axes, colors: Sequence[ColorValue], *,
    parameters: PlotParameters | None = None,
) -> list[Line2D]:
    """Update all ordinary line colors on an axes and refresh its legend.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    colors : sequence of Matplotlib colors
        One color per line in ``ax.lines``.
    parameters : PlotParameters, optional
        Settings for recreating an existing legend after updating colors.

    Returns
    -------
    list of matplotlib.lines.Line2D
        Modified line artists.

    Raises
    ------
    ValueError
        If the number of colors does not match the number of lines.
    """
    lines = list(ax.lines)
    if len(colors) != len(lines):
        raise ValueError("Exactly one color is required for each line.")
    for line, color in zip(lines, colors, strict=True):
        line.set_color(color)
    if ax.get_legend() is not None:
        add_legend(ax, parameters)
    return lines


def add_colorbar(
    mappable: ScalarMappable,
    *,
    ax: Axes,
    parameters: PlotParameters | None = None,
    label: str | None = None,
    location: Literal["right", "left", "top", "bottom"] = "right",
    fraction: float = 0.05,
    pad: float = 0.04,
    spacing: Literal["uniform", "proportional"] = "uniform",
    draw_edges: bool = False,
) -> Colorbar:
    """Attach a colorbar to a mappable artist on an axes.

    Parameters
    ----------
    mappable : matplotlib.cm.ScalarMappable
        An image, heatmap or histogram artist to use for color scaling.
    ax : matplotlib.axes.Axes
        Axes to which the colorbar belongs.
    parameters : PlotParameters, optional
        Typography and default colorbar label.
    label : str, optional
        Colorbar label overriding ``parameters.colorbar_label``.
    location : {"right", "left", "top", "bottom"}
        Colorbar placement.
    fraction, pad : float
        Axes fraction and separation used by Matplotlib.
    spacing : {"uniform", "proportional"}
        Interval layout for a discretized colorbar.
    draw_edges : bool
        Draw boundaries between color intervals.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The created colorbar.
    """
    options = parameters or PlotParameters()
    cb = ax.figure.colorbar(
        mappable, ax=ax, location=location, fraction=fraction,
        pad=pad, spacing=spacing, drawedges=draw_edges,
    )
    label_text = options.colorbar_label if label is None else label
    if label_text is not None:
        cb.set_label(label_text, fontsize=options.fontsize)
    cb.ax.tick_params(labelsize=options.tick_size)
    return cb


def format_text(
    text: Text, *, font_size: float | None = None,
    weight: str | int | None = None,
    style: str | None = None,
    family: str | None = None,
    color: ColorValue | None = None,
    background_color: ColorValue | None = None,
    alpha: float | None = None,
) -> Text:
    """Modify selected attributes on a Matplotlib text artist.

    Parameters
    ----------
    text : matplotlib.text.Text
        Existing text object to modify.
    font_size : float, optional
        New font size in points.
    weight, style, family : str, optional
        Font weight, style and family specifications.
    color, background_color : Matplotlib color, optional
        Foreground and background colors.
    alpha : float, optional
        Text opacity between zero and one.

    Returns
    -------
    matplotlib.text.Text
        The same modified text object.
    """
    if font_size is not None:
        text.set_fontsize(font_size)
    if weight is not None:
        text.set_weight(weight)
    if style is not None:
        text.set_style(style)
    if family is not None:
        text.set_family(family)
    if color is not None:
        text.set_color(color)
    if background_color is not None:
        text.set_backgroundcolor(background_color)
    if alpha is not None:
        text.set_alpha(alpha)
    return text


def add_figure_text(
    figure: Figure, text: str, *, x: float = 0.5, y: float = 0.95,
    font_size: float = 14, highlighted: bool = False,
    **kwargs: Any,
) -> Text | Any:
    """Add figure-level text, optionally using ``highlight_text`` markup.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Target figure.
    text : str
        Literal text or ``highlight_text`` markup if highlighted is True.
    x, y : float
        Figure-relative text coordinates.
    font_size : float
        Text font size in points.
    highlighted : bool, default False
        Use optional third-party ``highlight_text`` text markup.
    **kwargs
        Further Matplotlib or ``highlight_text.fig_text`` options.

    Returns
    -------
    matplotlib.text.Text or highlight_text.HighlightText
        Created text artist or highlight-text wrapper.

    Raises
    ------
    ImportError
        If highlighted text is requested and the optional package is missing.
    """
    if highlighted:
        try:
            from highlight_text import fig_text
        except ImportError as exc:
            raise ImportError("Install 'highlight_text' for highlighted figure text.") from exc
        # fig_text targets the supplied figure, not an implicit current plot.
        return fig_text(x=x, y=y, s=text, fig=figure, fontsize=font_size, **kwargs)
    return figure.text(x, y, text, fontsize=font_size, **kwargs)


def compute_limits(
    values: Iterable[float] | NDArray[Any], *, margin: float = 0.1
) -> tuple[float, float]:
    """Compute padded minimum and maximum values for numeric data.

    Parameters
    ----------
    values : iterable of float or numpy.ndarray
        Numeric observations, including nested NumPy arrays.
    margin : float, default 0.1
        Fraction of data span added at both ends.

    Returns
    -------
    tuple of float
        Lower and upper plotting bounds.

    Raises
    ------
    ValueError
        If margin is negative or data contain no finite values.
    """
    if margin < 0:
        raise ValueError("'margin' must be non-negative.")
    data = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        raise ValueError("At least one finite numeric value is required.")
    low, high = float(valid.min()), float(valid.max())
    span = high - low or max(abs(low), 1.0)
    padding = margin * span
    return low - padding, high + padding


def set_figure_size(
    figure: Figure, width: float, height: float, *,
    unit: Literal["in", "cm"] = "cm",
) -> Figure:
    """Resize a figure using inches or centimeters.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to resize.
    width, height : float
        Positive dimensions.
    unit : {"in", "cm"}, default "cm"
        Unit of the dimensions.

    Returns
    -------
    matplotlib.figure.Figure
        Resized figure.
    """
    if width <= 0 or height <= 0:
        raise ValueError("'width' and 'height' must be positive.")
    if unit not in ("in", "cm"):
        raise ValueError("'unit' must be 'in' or 'cm'.")
    multiplier = 1.0 if unit == "in" else 1 / 2.54
    figure.set_size_inches(width * multiplier, height * multiplier)
    return figure


def save_figure(
    figure: Figure,
    path: str | Path,
    *,
    dpi: int = 300,
    file_format: str | None = None,
    size_cm: tuple[float, float] | None = None,
    create_directories: bool = False,
    tight: bool = True,
    close: bool = False,
) -> Path:
    """Export a figure without swallowing filesystem or backend errors.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to save.
    path : str or pathlib.Path
        Output path. If no extension exists, '.png' is appended.
    dpi : int, default 300
        Raster image resolution.
    file_format : str, optional
        Export format such as 'png', 'pdf', 'svg' or 'eps'. If specified,
        this determines/replaces the file extension.
    size_cm : tuple of float, optional
        Resize the figure to this physical width and height before export.
    create_directories : bool, default False
        Create missing output directories.
    tight : bool, default True
        Use a tight bounding box when exporting.
    close : bool, default False
        Close the figure after successful export.

    Returns
    -------
    pathlib.Path
        Absolute or relative path of the exported file.

    Raises
    ------
    FileNotFoundError
        If the parent directory does not exist and creation is disabled.
    ValueError
        If the output resolution is not positive.
    """
    if dpi < 1:
        raise ValueError("'dpi' must be positive.")
    output = Path(path)
    if file_format is not None:
        output = output.with_suffix("." + file_format.lower().lstrip("."))
    elif not output.suffix:
        output = output.with_suffix(".png")
    if create_directories:
        output.parent.mkdir(parents=True, exist_ok=True)
    elif not output.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output.parent}")
    if size_cm is not None:
        set_figure_size(figure, *size_cm, unit="cm")
    figure.savefig(output, dpi=dpi, bbox_inches="tight" if tight else None)
    if close:
        plt.close(figure)
    return output


def show_figure(figure: Figure, *, block: bool = False) -> None:
    """Show a figure using the current interactive Matplotlib backend.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to display.
    block : bool, default False
        Whether to block until plot windows are closed.
    """
    if hasattr(figure, "number"):
        plt.figure(figure.number)
    plt.show(block=block)


def close_figure(figure: Figure | None = None) -> None:
    """Close a figure; close the active figure when none is supplied."""
    plt.close(figure)


def close_all_figures() -> None:
    """Close all open Matplotlib figures."""
    plt.close("all")


def update_layout(figure: Figure) -> Figure:
    """Apply tight layout to an existing figure and return it."""
    figure.tight_layout()
    return figure


def set_latex_text(enabled: bool = True) -> None:
    """Enable or disable external LaTeX rendering in Matplotlib globally.

    Notes
    -----
    When enabled, a working LaTeX installation is required for rendering.
    This function changes Matplotlib's global rcParams configuration.
    """
    plt.rcParams["text.usetex"] = enabled


def maximize_figure(figure: Figure) -> None:
    """Maximize a GUI figure window where the backend permits it.

    Raises
    ------
    RuntimeError
        If the backend is headless or has no window-maximization API.
    """
    manager = figure.canvas.manager
    window = getattr(manager, "window", None)
    if window is None:
        raise RuntimeError("This Matplotlib backend has no GUI window.")
    if hasattr(window, "showMaximized"):
        window.showMaximized()  # Qt backend.
    elif hasattr(window, "state"):
        window.state("zoomed")  # Tk backend on Windows.
    elif hasattr(manager, "full_screen_toggle"):
        manager.full_screen_toggle()
    else:
        raise RuntimeError("This Matplotlib backend does not support maximization.")


__all__ = [
    "create_figure", "configure_axes", "add_legend",
    "add_legend_with_subtitles", "update_line_colors", "add_colorbar",
    "format_text", "add_figure_text", "compute_limits",
    "set_figure_size", "save_figure", "show_figure", "close_figure",
    "close_all_figures", "update_layout", "set_latex_text", "maximize_figure",
]
