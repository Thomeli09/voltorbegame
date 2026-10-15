# VoltorbeGame — Voltorb Flip solver

[![Python](https://img.shields.io/badge/Python-3.15%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-GPL--3.0--or--later-blue)](LICENSE)

**VoltorbeGame** is an independent Python project that analyzes the **Voltorb Flip**
minigame in *Pokémon HeartGold and SoulSilver*. Given the points and Voltorb
counts indicated for each row and column, as well as tiles already revealed,
it estimates tile risks and recommends which hidden tile to flip next.

It includes **three selectable solver methods**, an interactive command-line
session, and reusable NumPy/Matplotlib visualization helpers adapted from the
GanttChart project.

> **Disclaimer:** This is a fan-made analysis tool, not affiliated with or
> endorsed by Nintendo, Game Freak, Creatures, or The Pokémon Company.
> Recommendations are statistical estimates, not guaranteed safe moves.

## Features

- Immutable, validated row/column clues (`PuzzleClues`).
- Mutable revealed-board state (`VoltorbGame`) with consistency checks.
- Constraint-based enumeration that counts valid boards without storing them all.
- Original row/column scoring logic, retained as a separate **legacy solver**.
- A fast approximate **heuristic solver** when exhaustive search is impractical.
- Exact per-cell distributions for tile values 0, 1, 2, and 3 where enumeration succeeds.
- Configurable recommendation rules and an explicit search-node budget.
- Reusable plotting helpers (`PlotParameters`, `plot_image`, `create_figure`,
  `save_figure`) and game-specific heatmaps.
- Importable API, interactive CLI, examples and automated tests.

## Requirements

- **Python 3.15 or later** (the declared supported version).
- NumPy and Matplotlib, installed automatically by pip.
- Development/testing tools are included in the optional `dev` extra.

## Installation

Clone the repository:

```bash
git clone https://github.com/Thomeli09/voltorbegame.git
cd voltorbegame
```

Create a virtual environment (example commands for Windows PowerShell):

```powershell
py -3.15 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

On Linux/macOS, use `python3.15 -m venv .venv` and `source .venv/bin/activate`.
Alternatively, `python -m pip install -r requirements.txt` installs the editable
package. To install development dependencies:

```bash
python -m pip install -e ".[dev]"
# Or: python -m pip install -r requirements-dev.txt
```

To run the tests:

```bash
python -m pytest -q
```

For a regular non-editable installation, use `python -m pip install .`.

## Repository layout

```text
src/voltorbe_game/
    __init__.py                # Public API
    puzzle.py                  # PuzzleClues and VoltorbGame board state
    solver.py                  # Shared solver API and exact enumeration
    solvers/
        __init__.py
        legacy.py              # Original CmptScores-style ranking
    simulation/
        __main__.py            # python -m voltorbe_game.simulation
        run.py                 # CLI and interactive session
    utils/                     # Reused generic data tools
    visualization/
        plot_parameters.py     # Shared plotting configuration
        plot_basic.py          # Figures and formatting
        plot_2d.py             # General 2D plot helpers
        plot_board.py          # Voltorb-specific heatmaps
examples/                       # Runnable scripts
    01_basic_solver.py
    02_visualize_and_update.py
    03_small_puzzle_and_fallback.py
    04_compare_solvers.py
    05_compare_solver_plots.py
tests/                          # Automated tests
```

## Quick start

```python
from voltorbe_game import PuzzleClues, VoltorbGame, solve

clues = PuzzleClues(
    column_points=[5, 7, 3, 5, 8],
    column_voltorbs=[2, 1, 3, 1, 0],
    row_points=[5, 8, 5, 5, 5],
    row_voltorbs=[2, 1, 0, 2, 2],
)
game = VoltorbGame(clues)

result = solve(game, method="exact")
print("Valid boards:", result.possible_grids)
print("Recommended flip:", result.recommendation)
```

For the above clues, **828 boards** satisfy the row/column constraints. The
exact risk-first policy recommends **row 2, column 5** (one-based coordinates),
with zero modeled Voltorb risk and about **73.6%** multiplier chance.

The `row_*` clue vectors correspond to your original `Y` data, and the
`column_*` vectors correspond to the original `X` data. The **Python API uses
zero-based indices**, while the terminal display uses **one-based indices**.

### Reveal a tile and recompute

```python
recommendation = result.recommendation
if recommendation is not None:
    # Enter only the value you actually observed in the game.
    observed_value = 2  # illustrative, NOT a prediction
    game.reveal(recommendation.row, recommendation.column, observed_value)
    result = solve(game, method="exact")
```

Unknown cells are stored as `-1`, Voltorbs as `0`, and revealed safe cells as
`1`, `2`, or `3`. A game is won when all multiplier tiles (values 2 and 3) are
revealed; a revealed Voltorb ends the game.

## Choosing a solver

`solve()` accepts a `method` argument. All methods return the same
`SolverResult` structure so you can change methods without rewriting plotting
or simulation code.

| `method` | How it works | Returned analysis | Recommendation |
|---|---|---|---|
| `"exact"` **(default)** | Counts all constraint-compatible boards, accumulating each tile value's frequency | `mode="exact"` | Lowest risk among tiles with multiplier potential (`risk_first`), or an approximate risk/reward ranking (`expected_gain`) |
| `"legacy"` | Uses the exact enumeration frequencies **plus your original row/column score** from `CmptScores()` | `mode="exact"` | Forced-safe tiles first, then minimum risk among eligible positive-score tiles; original row-major tie behavior |
| `"heuristic"` | Estimates risk and multiplier potential from the remaining row and column clues; **no enumeration** | `mode="heuristic"` | Approximate risk-first or expected-gain ranking |

```python
from voltorbe_game import solve
from voltorbe_game.simulation import example_game

game = example_game()

exact = solve(game, method="exact", strategy="risk_first")
legacy = solve(game, method="legacy")
quick = solve(game, method="heuristic")

# Alternative selection rule for the exact solver:
reward_oriented = solve(game, method="exact", strategy="expected_gain")
```

**Legacy behavior:** The legacy scorer is implemented separately in
`voltorbe_game/solvers/legacy.py`. It recreates the old line-score formula,
including infinite scores for lines with no remaining Voltorbs. Unlike the
original code, it uses the new memory-efficient enumeration engine and rejects
already revealed tiles. It does *not* retain the older `input()`-based solver
loop or store all plausible grids in memory.

On the example clues, `legacy` recommends **row 1, column 5** (also zero modeled
Voltorb risk), compared with **row 2, column 5** under the exact risk-first
policy. The different recommendation reflects a **different ranking rule**;
the exact probability matrices themselves are the same.

### Exact results, heuristics and limits

Use `result.mode` to distinguish real constraint frequencies from approximations:

```python
result = solve(game, method="exact", max_nodes=2_000_000, fallback=True)
print(result.method)          # Requested solver: 'exact'
print(result.mode)            # 'exact' or 'heuristic' if fallback occurred
print(result.possible_grids)  # int for exact, None for heuristic
```

When an exact/legacy search reaches `max_nodes`, the default is to raise
`SearchLimitExceeded`; setting `fallback=True` explicitly permits a heuristic
result. In that case, `method` still records your requested solver but
`mode="heuristic"` and `possible_grids=None`. `method="legacy"` uses the original
line scores in its fallback, while `method="exact"` uses the faster default
heuristic ranking.

**Important:** Enumeration considers every compatible board equally likely,
which may not reproduce HeartGold/SoulSilver's level-dependent generation
rules. Thus "exact" means **exact under the puzzle constraints and equal-board
assumption**, not proven in-game probabilities. The other methods' risk
estimates are **uncalibrated** and must not be presented as true probabilities.
The solver does not optimize the full sequence of future reveals or guarantee
that a safe path to victory exists.

### Result fields

| Attribute | Meaning |
|---|---|
| `result.method` | Selected solver: `exact`, `legacy`, `heuristic` |
| `result.mode` | Actual evaluation: `exact` or `heuristic` |
| `result.strategy` | Ranking policy: `risk_first`, `expected_gain`, `legacy_score` |
| `result.possible_grids` | Valid-board count, or `None` for heuristic mode |
| `result.probabilities` | `(rows, columns, max_value+1)` exact frequency array, or `None` |
| `result.voltorb_risk` | Probability (exact) or heuristic risk estimate, range 0–1 |
| `result.multiplier_chance` | Exact chance of tile ≥2, or `None` in heuristic mode |
| `result.scores` | Exact reward score, legacy line score, or heuristic potential, depending on method |
| `result.recommendation` | `FlipRecommendation(row, column, ..., strategy)` or `None` |

`result.voltorb_risk` contains `NaN` on revealed tiles. Some legacy scores can
be `inf`, indicating cells identified as safe from one of their line clues.
`strategy="legacy_score"` is valid only with `method="legacy"`.

## Visualizations

```python
from voltorbe_game import solve
from voltorbe_game.simulation import example_game
from voltorbe_game.visualization import PlotParameters, plot_analysis, save_figure

state = example_game()
result = solve(state, method="legacy")

style = PlotParameters(figsize=(15, 5), fontsize=11)
fig, axes = plot_analysis(state, result, parameters=style)
save_figure(fig, "voltorb_legacy.png")
```

The three-panel analysis shows the revealed board, modeled Voltorb risk and
multiplier opportunity. In legacy mode, the last panel shows the **original
row/column scores**, including `∞` for forced-safe cells. A red outline marks
the recommended flip.

Generic plotting functions are reused from the GanttChart project but bundled
locally to keep this package independent. To avoid duplicating them across
future projects, consider extracting them into a standalone shared package.

## Command-line interface

Install the package, then run:

```bash
# One recommendation with the exact solver
python -m voltorbe_game.simulation --once

# Original scoring policy + exact probabilities
python -m voltorbe_game.simulation --method legacy --once

# No enumeration (fast approximate estimates)
python -m voltorbe_game.simulation --method heuristic --once

# Show Matplotlib figures after each interaction
python -m voltorbe_game.simulation --method exact --plot

# Allow a labeled fallback when the search exceeds the budget
python -m voltorbe_game.simulation --method legacy --fallback --max-nodes 100000
```

You can also use the installed `voltorbe-game` command with the same arguments.
The interactive CLI reads the **actual observed value** of each selected tile
and updates the board for the next analysis.

## Examples and tests

```bash
python examples/01_basic_solver.py
python examples/02_visualize_and_update.py
python examples/03_small_puzzle_and_fallback.py
python examples/04_compare_solvers.py
python examples/05_compare_solver_plots.py
python -m pytest -q
```

The files in `examples/` are independent scripts and demonstrate both
calculation and visualization using the package's public API.

## Citation

If you use VoltorbeGame in academic, technical or published work, please
cite the software using [`CITATION.cff`](CITATION.cff).

```bibtex
@software{Thommes_VoltorbeGame_2026,
  author       = {Thommes, Eliott},
  title        = {VoltorbeGame: A Python solver and visualization library for Voltorb Flip in Pokémon HeartGold and SoulSilver},
  year         = {2026},
  version      = {0.0.1},
  url          = {https://github.com/Thomeli09/voltorbegame}
}
```

Author: **Eliott Thommes** · [ORCID 0009-0005-0697-2438](https://orcid.org/0009-0005-0697-2438)

## License

Copyright © 2026 Eliott Thommes.

Licensed under the **GNU General Public License, version 3 or later**
(`GPL-3.0-or-later`). See [`LICENSE`](LICENSE) for the complete GPL-3.0 text.