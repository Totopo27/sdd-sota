# litreview

Literature Review tools for Master's Thesis in Electrical Engineering at University of Costa Rica.

A structured Python package for corpus exploration: fetch papers from Zotero, discover topics with BERTopic, validate with zero-shot classification, and generate statistics and visualizations.

## Project Structure

```
litreview/
├── src/litreview/          — Core Python package (src-layout)
│   ├── fetchers/           — Data fetchers (Zotero)
│   ├── analyzers/          — Topic modeling & zero-shot classification
│   ├── cli/                 — CLI entry points
│   ├── pipeline.py         — ReviewPipeline orchestrator
│   ├── statistics.py       — Corpus statistics & gap analysis
│   ├── visualization.py    — Plotting functions
│   ├── config.py           — Configuration dataclasses
│   └── __init__.py         — Public API
├── tests/                  — Unit tests
├── config.yaml             — Central configuration
├── .env-example            — Environment variable template
└── pyproject.toml          — Dependencies & build config
```

## Setup

This project uses `pyproject.toml` for dependency management with [`uv`](https://docs.astral.sh/uv/) as the package manager. Requires Python >= 3.12.

```bash
uv sync          # Install dependencies and create uv.lock
uv pip install -e .  # Install package in development mode
```

## Usage

### 1. Configure

Copy `.env-example` to `.env` and fill in your Zotero credentials:

```bash
cp .env-example .env
```

Edit `config.yaml` to set models, labels, and paths.

### 2. Run Analysis

```bash
litreview-analysis --config config.yaml --fetch-zotero
```

Or with a local CSV:

```bash
litreview-analysis --config config.yaml --input papers.csv
```

### 3. Generate Plots

```bash
litreview-plots --config config.yaml
```

### 4. Programmatic Usage

```python
from litreview import ReviewPipeline, load_config

config = load_config("config.yaml")
pipeline = ReviewPipeline(config)
report = pipeline.run()

# Export results
report.export_csv("results/")
report.export_json("results/")
report.generate_plots("results/plots/")
print(report.summary())
```

## Configuration

### Environment Variables

| Variable | Description |
|---|---|
| `ZOTERO_LIBRARY_ID` | Your Zotero library/user ID |
| `ZOTERO_API_KEY` | Your Zotero API key |

### config.yaml

```yaml
zotero:
  library_type: user
  collection_name: my_collection  # optional

bertopic:
  embedding_model: all-MiniLM-L6-v2
  min_topic_size: 10
  seed_topics: [["deep learning"], ["optimization"]]  # optional

zeroshot:
  models:
    - facebook/bart-large-mnli
    - typeform/distilbert-base-uncased-mnli
  threshold: 0.5
  candidate_labels:
    A: Label A
    B: Label B

paths:
  plots: results/plots
```

## Dependencies

Defined in `pyproject.toml` and resolved by `uv`. See `uv.lock` for exact pinned versions.

- **pandas, numpy** — Data manipulation
- **matplotlib** — Visualization
- **transformers, torch** — Zero-shot classification
- **datasets** — HuggingFace dataset utilities
- **PyYAML** — Configuration parsing
- **pyzotero** — Zotero API client
- **python-dotenv** — Environment variable loading
- **bertopic** — Topic modeling
- **sentence-transformers** — Text embeddings
- **umap-learn, hdbscan** — Dimensionality reduction & clustering

## Development

```bash
uv run pytest          # Run tests
uv run ruff check .    # Lint
```
