# Project Website

This folder contains a lightweight static website for presenting the project to an instructor or professor.

## Run Locally

From the project root:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/site/
```

The page reads live data from:

- `artifacts/metrics.json`
- `artifacts/model_leaderboard.csv`
- `artifacts/sample_predictions.csv`
- `artifacts/figures/*.png`
