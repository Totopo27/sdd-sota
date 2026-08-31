# litreview

Literature Review tools for Master's Thesis in Electrical Engineering at University of Costa Rica.

## Project Structure

- `litreview/`: Core Python package containing analysis and visualization logic.
- `config.yaml`: Central configuration file for models, labels, sources, and paths.
- `scripts/`: Entry point scripts for the pipeline.
- `analysis.slurm`: SLURM script for running the analysis on a cluster.

## Setup

This project uses `pyproject.toml` for dependency management with [`uv`](https://docs.astral.sh/uv/) as the package manager.

```bash
uv sync          # Install dependencies and create uv.lock
uv run python scripts/run_analysis.py --config config.yaml
```

`uv sync` resolves dependencies and writes `uv.lock` for reproducible builds. Use `uv run` to execute scripts with the project's environment.

## Usage

### 1. Run Analysis
To process data and run the zero-shot classification:

**Using local CSVs:**
```bash
uv run python scripts/run_analysis.py --config config.yaml
```

**Using Zotero Library:**
1. Configure your Zotero credentials in `config.yaml`.
2. Run:
```bash
uv run python scripts/run_analysis.py --config config.yaml --fetch-zotero
```

*Note: Classification requires a GPU and may take significant time.*

### 2. Generate Plots
To generate visualizations and filter the classified data:

```bash
uv run python scripts/run_plots.py --config config.yaml
```

## Dependencies

Defined in `pyproject.toml` and resolved by `uv`. See `uv.lock` for exact pinned versions.
