"""Typed and reusable configuration for Matplotlib plotting.

The configuration object stores *settings*, not Matplotlib artists.  Figures,
axes, lines and colorbars are managed explicitly by the drawing functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

import numpy as np
from matplotlib import colormaps, colors
from matplotlib.colors import Colormap


# These aliases use modern Python type-alias declarations (Python >= 3.12).
type ColorValue = str | tuple[float, float, float] | tuple[float, float, float, float]
type AxisValue = int | float | str | date | datetime | np.datetime64
type AxisLimits = tuple[AxisValue, AxisValue]
type GridAxis = Literal["x", "y", "both"]
type LegendLocation = str | int


@dataclass(slots=True, kw_only=True)
class PlotParameters:
    """Store general-purpose plotting options without creating a figure.

    Parameters
    ----------
    color : str or RGB/RGBA tuple, optional
        Default color used when drawing a single artist. If omitted,
        Matplotlib's normal color cycle is used.
    palette : sequence of colors, optional
        Colors available to plots containing multiple artists. The palette
        is selected by index rather than consumed by reading a property.
    colormap : str or matplotlib.colors.Colormap, default "viridis"
        Colormap used by density, histogram and image plots.
    linestyle : str, default "-"
        Matplotlib line-style specification.
    linewidth : float, default 1.5
        Width of lines in points.
    marker : str, optional
        Matplotlib marker specification.
    markersize : float, default 6
        Marker size in points.
    alpha : float, default 1
        Artist opacity, between zero and one.
    hatch : str, optional
        Matplotlib hatch pattern for polygon artists.
    title, xlabel, ylabel : str, optional
        Axis title and labels.
    fontsize, title_size, tick_size, legend_size : float
        General, title, tick and legend font sizes in points.
    show_legend : bool, default True
        Whether ``configure_axes`` should create a legend when labeled
        artists are present.
    legend_location : str or int, default "best"
        Matplotlib location argument for legends.
    legend_outside : bool, default False
        Place the legend to the right of the axes.
    show_grid : bool, default False
        Whether to draw the grid.
    grid_axis : {"x", "y", "both"}, default "both"
        Axis on which the grid is displayed.
    grid_color, grid_linestyle, grid_linewidth, grid_alpha
        Matplotlib grid styling parameters.
    xlim, ylim : tuple, optional
        Explicit axis limits, including numerical or date-like limits.
    xscale, yscale : str, default "linear"
        Matplotlib axis scale types.
    date_format : str, optional
        Strftime-style major tick format for a date x-axis.
    rotate_dates : bool, default False
        Rotate x-axis date labels to reduce overlap.
    show_box : bool, default True
        Whether the four axes spines should be visible.
    aspect : str or float, optional
        Aspect ratio passed to ``Axes.set_aspect``.
    figsize : tuple of float, default (9, 5)
        Figure width and height in inches.
    colorbar_label : str, optional
        Default label for colorbars.

    Notes
    -----
    No figure or axes references are stored here. This makes a configuration
    safe to reuse on many separate figures, including subplot grids.
    """

    color: ColorValue | None = None
    palette: tuple[ColorValue, ...] = field(default_factory=tuple)
    colormap: str | Colormap = "viridis"
    linestyle: str = "-"
    linewidth: float = 1.5
    marker: str | None = None
    markersize: float = 6.0
    alpha: float = 1.0
    hatch: str | None = None
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    fontsize: float = 12.0
    title_size: float = 14.0
    tick_size: float = 11.0
    legend_size: float = 11.0
    show_legend: bool = True
    legend_title: str | None = None
    legend_location: LegendLocation = "best"
    legend_outside: bool = False
    show_grid: bool = False
    grid_axis: GridAxis = "both"
    grid_color: ColorValue | None = None
    grid_linestyle: str = "-"
    grid_linewidth: float = 0.5
    grid_alpha: float = 0.4
    xlim: AxisLimits | None = None
    ylim: AxisLimits | None = None
    xscale: str = "linear"
    yscale: str = "linear"
    date_format: str | None = None
    rotate_dates: bool = False
    show_box: bool = True
    aspect: str | float | None = None
    figsize: tuple[float, float] = (9.0, 5.0)
    colorbar_label: str | None = None

    def __post_init__(self) -> None:
        """Validate parameters that have unambiguous numerical bounds."""
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("'alpha' must be between 0 and 1.")
        if not 0.0 <= self.grid_alpha <= 1.0:
            raise ValueError("'grid_alpha' must be between 0 and 1.")
        for name in (
            "linewidth", "markersize", "fontsize", "title_size",
            "tick_size", "legend_size", "grid_linewidth",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"'{name}' must be non-negative.")
        if self.grid_axis not in ("x", "y", "both"):
            raise ValueError("'grid_axis' must be 'x', 'y' or 'both'.")
        if len(self.figsize) != 2 or any(value <= 0 for value in self.figsize):
            raise ValueError("'figsize' must contain two positive lengths.")
        if self.color is not None and not colors.is_color_like(self.color):
            raise ValueError(f"Invalid Matplotlib color: {self.color!r}.")
        if any(not colors.is_color_like(item) for item in self.palette):
            raise ValueError("All palette entries must be valid Matplotlib colors.")
        if isinstance(self.colormap, str):
            colormaps.get_cmap(self.colormap)  # Fail early for unknown names.
        elif not isinstance(self.colormap, Colormap):
            raise TypeError("'colormap' must be a name or Matplotlib Colormap.")

    def color_at(self, index: int = 0) -> ColorValue | None:
        """Return the color at an index without modifying the palette.

        Parameters
        ----------
        index : int, default 0
            Zero-based index of the desired color. Indices wrap around.

        Returns
        -------
        ColorValue or None
            Palette color, the default color, or None for Matplotlib's cycle.
        """
        if self.palette:
            return self.palette[index % len(self.palette)]
        return self.color

    def sample_colormap(
        self, count: int, *, start: float = 0.0, stop: float = 1.0
    ) -> list[tuple[float, float, float, float]]:
        """Generate evenly spaced RGBA colors from the configured colormap.

        Parameters
        ----------
        count : int
            Number of colors to generate; must be non-negative.
        start, stop : float
            Inclusive colormap positions between zero and one.

        Returns
        -------
        list of tuple of float
            RGBA color tuples.

        Raises
        ------
        ValueError
            If count is negative or bounds are outside [0, 1].
        """
        if count < 0:
            raise ValueError("'count' must be non-negative.")
        if not (0 <= start <= stop <= 1):
            raise ValueError("'start' and 'stop' must satisfy 0 <= start <= stop <= 1.")
        cmap = colormaps.get_cmap(self.colormap)
        return [tuple(map(float, cmap(v))) for v in np.linspace(start, stop, count)]


__all__ = ["PlotParameters", "ColorValue", "AxisValue", "AxisLimits", "GridAxis", "LegendLocation"]
