# voltorbe_game: reusable visualization modules

This package fragment updates your three Matplotlib helper modules for modern Python
(compatible with Python 3.15 syntax) and adds a public `visualization/__init__.py`.

Place the modules in your existing project:

```text
src/voltorbe_game/visualization/
    __init__.py
    plot_parameters.py
    plot_basic.py
    plot_2d.py
```

The modules depend on `matplotlib` and `numpy`. **`highlight_text` is optional**, needed
only when using `add_figure_text(..., highlighted=True)`. `pytest` is only used in tests.

## Usage

```python
import numpy as np
from voltorbe_game.visualization import (
    PlotParameters, create_figure, plot_line,
    configure_axes, save_figure,
)

style = PlotParameters(
    title="Progress", xlabel="Day", ylabel="Completion",
    color="steelblue", marker="o", linewidth=2,
    show_grid=True, figsize=(8, 4),
)
fig, ax = create_figure(parameters=style)
line = plot_line(ax, [0, 1, 2, 3], [0, 0.2, 0.6, 1.0],
                 parameters=style, label="Project")
configure_axes(ax, style)
line.set_marker("s")  # Change the artist after plotting, if desired.
save_figure(fig, "progress.pdf")
```

### Design conventions

- `PlotParameters` stores appearance settings. It does not own artists, figures or axes.
- All drawing functions receive an explicit `Axes` argument and return Matplotlib artists.
- `create_figure` returns a figure and one `Axes` or an array of axes.
- `configure_axes` is normally called after drawing, so legends detect plotted labels.
- No function calls `plt.show()` implicitly. Use `show_figure(fig)` when needed.
- Colors in `palette` are chosen by index; reading a color never consumes it.
- File-saving errors and invalid values raise exceptions rather than printing warnings.

### Migration table

| Old method/function | New API |
|---|---|
| `ParamPLT` | `PlotParameters` |
| `DefaultParamPLT` | `PlotParameters()` |
| `ColorFillList` | `PlotParameters.sample_colormap` |
| `ColorFullList`, `HatchFullList` | `palette` and `hatch` attributes |
| `StartPlots`, `PLTMultiPlot` | `create_figure(rows, columns)` |
| `PLTShow` | `configure_axes(ax, style)` + `show_figure(fig)` |
| `PLTGrid`, `PLTLimit`, `PLTScaleType`, `PLTDate`, `PLTBox` | `configure_axes(ax, style)` |
| `PLTTitleAxis`, `PLTSizeAxis` | `configure_axes(ax, style)` |
| `PLTTitleModified` | `add_figure_text` |
| `PLTLegend` | `add_legend` |
| `PLTLegendWithTitlesSubtitles` | `add_legend_with_subtitles` |
| `UpdatePlotColorsAndLegend` | `update_line_colors` |
| `PLTColorBar` | `add_colorbar(mappable, ax=ax)` |
| `PLTCmptLimit` | `compute_limits` |
| `PLTScreenSize`, `PLTSetWindowSize` | `set_figure_size` |
| `PLTSave` | `save_figure` |
| `PLTUpdateLayout` | `update_layout` |
| `PLTScreenMaximize` | `maximize_figure` (GUI backend required) |
| `PLTLatexStyle` | `set_latex_text` (LaTeX installation required) |
| `ClosePlot`, `CloseAllPlots` | `close_figure`, `close_all_figures` |
| `ClosePlotsOnDemand` | caller handles `input()`; then `close_all_figures()` |
| `PLTShowRefSavePlace` | `Path.cwd()` |
| `FormatText` | `format_text` |
| `PLTText` | `plot_text` |
| `PLTPlot`, `PLTEmptyPlot`, `PLTUpdatePlot` | `plot_line`, `plot_empty_line`, `update_line` |
| `PLTPlotSeries` | `plot_series` |
| `PLTVHLine` | `draw_reference_line` |
| `PLTAXLine` | `draw_infinite_line` |
| `PLTCoordsSpan` | `draw_span` |
| `PLTFill` | `fill_area` |
| `PLTBar` | `plot_bars` |
| `PLTHist`, `PLTHist2D` | `plot_histogram`, `plot_histogram2d` |
| `PLTPie` | `plot_pie` |
| `PLTImShow` | `plot_image` |
| `PLT2DCircle` | `draw_circle` |

### Deliberate API changes

Existing calls to old `PLT*` functions are **not source-compatible** with these new
functions. The 2D plotting helpers no longer use global `plt.gca()` or consume
stored colors, legend labels, hatches, and limits as queues. Matplotlib's standard
string values (e.g. `"viridis"`, `"--"`, `"log"`) replace old integer lookup codes.
To customize a figure, use the returned Matplotlib artists. The old options for 3D
plots and modal `input()` prompts are not carried forward into these *2D* modules.

### Test

```bash
pip install pytest numpy matplotlib
pytest tests/test_visualization.py
```

Syntax and runtime tests were run using Python 3.13 and Matplotlib 3.10.8; this
package fragment has not been runtime-tested on Python 3.15.
