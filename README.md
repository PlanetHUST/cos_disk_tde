# TDE Rate Disk Workflow

This repository runs a dynamical NSC-AGN tidal disruption event (TDE) rate
workflow.  The main calculation samples a grid of SMBH and nuclear star cluster
(NSC) masses, repeats a Monte Carlo realization at each grid point, computes
damping, migration, relaxation, and delivery timescales, and writes CSV/JSON
outputs plus diagnostic plots.

The current implementation lives under `src/tde_rate`.  The root `main.py`
script is a thin command-line wrapper around that package.

## Repository Layout

```text
main.py                     Primary command-line entry point
test.py                     Backward-compatible CLI wrapper
src/tde_rate/
  cli.py                    Argument parser and TDEDynConfig construction
  config.py                 Configuration, constants, and output filenames
  engine.py                 One-realization physical workflow
  workflow.py               SMBH/NSC grid loop, checkpoints, summaries
  workers.py                Multiprocessing boundary for damping tasks
  outputs.py                CSV, metadata, statistics, and plots
  models.py                 Typed payload and result containers
tests/                      Unit and regression tests
doc/                        Local notes and rendered analysis files
```

Several historical physics support modules remain at the repository root
(`constants.py`, `disk.py`, `damp.py`, `sampling.py`, `helper_function.py`,
`general_class.py`, and `opacity.py`).  They are used by the package code and
preserve existing public names.

## Setup

Use a Python environment with the scientific stack installed:

```bash
python -m pip install numpy pandas matplotlib scipy
```

The project is currently run from the repository root rather than from an
installed package.  If matplotlib cannot write its default cache directory on a
cluster or sandbox, set a writable cache path before running commands:

```bash
export MPLCONFIGDIR=/tmp/matplotlib-cache
```

## Run a Simulation

Show the live command-line contract:

```bash
python main.py --help
```

Run a small grid:

```bash
python main.py 3 2 ./tde_dyn_results 4 --no-debug-results
```

The positional arguments are:

```text
num_samples        Mass-grid samples per axis; grid size is num_samples^2
num_repeats        Monte Carlo repeats per grid point
output_dir         Directory for CSV, JSON, and plot outputs
damping_cpu_count  Worker count; omitted means scheduler/local auto-detect
```

Useful options include:

```text
--duty-cycle-myr
--smbh-min-msun / --smbh-max-msun
--nsc-min-msun / --nsc-max-msun
--metallicity
--no-resupply
--resample-copy-orbits
--direct-tde
--tde-radius-rg
--checkpoint-interval
--no-visualizations
--plot-only [PATH]
```

Regenerate plots from an existing output directory without rerunning the
simulation:

```bash
python main.py --plot-only ./tde_dyn_results
```

## Programmatic Use

```python
from src.tde_rate import TDEDynConfig, run_simulation

config = TDEDynConfig(
    num_samp=3,
    num_repeats=2,
    output_dir="./tde_dyn_results",
    print_debug_results=False,
)

results = run_simulation(config)
```

For reproducible runs, specify important physical parameters explicitly.  The
CLI defaults and direct `TDEDynConfig()` defaults are not identical for every
field.

## Outputs

The output directory contains:

```text
tde_dyn_averaged_results.csv     Averaged rates for each SMBH/NSC grid point
timescales.csv                   Per-object damping, migration, relaxation,
                                 and delivery records
simulation_metadata.json         Configuration and runtime statistics
visualizations/                  Rate, heatmap, and timescale histogram PNGs
```

Column names encode units where practical.  Names ending in `_msun` are in
solar masses, `_myr` in Myr, and `_cm` in centimeters.

## Analysis Report

The local R Markdown report reads the generated CSV/JSON outputs:

```bash
Rscript -e 'rmarkdown::render("doc/tde_dyn_results_analysis.Rmd")'
```

It expects the default result directory `./tde_dyn_results` unless a different
`result_dir` parameter is supplied.  The report uses MathJax for LaTeX-style
formulae in HTML table labels and text.

## Validation

Run the regression tests:

```bash
python -m unittest discover -s tests -v
```

Check Python syntax/import compilation:

```bash
python -m compileall -q .
```

A minimal stochastic smoke run is possible but more expensive than the tests:

```bash
python main.py 1 1 /tmp/tde_dyn_smoke 1 --no-debug-results
```
