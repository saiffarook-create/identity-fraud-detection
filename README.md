# Identity Theft Detection

A beginner-friendly machine learning project for detecting identity theft, fraud, and account takeover attempts from tabular data.

This repository is designed to match a university cybersecurity ML project brief while staying realistic for a solo student. It includes:

- reproducible preprocessing
- simple feature engineering
- training and comparison of 3 classical ML models
- evaluation focused on imbalanced classification
- saved artifacts and figures
- a CLI for batch predictions
- notebooks for EDA, training, and results analysis

## Project Structure

```text
project-root/
├── artifacts/
├── config/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── evaluation/
│   ├── features/
│   ├── inference/
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   └── utils/
└── tests/
```

## Recommended Dataset

Primary target dataset:

- `Identity Fraud Detection` from Kaggle

Place your CSV in `data/raw/`, for example:

- `data/raw/identity_fraud.csv`

If your dataset uses a different filename or target column, pass them as CLI arguments.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Train The Models

```bash
python -m src.training.train --data data/raw/identity_fraud.csv --target is_fraud
```

## Move From Demo Data To Real Kaggle Data

1. Download the Kaggle identity-fraud CSV manually.
2. Put it inside `data/raw/`.
3. Inspect it with:

```bash
python -m src.preprocessing.inspect_dataset --data data/raw/your_file.csv
```

4. Open the generated config at `config/detected_dataset_config.json`.
5. Fix the target column if needed.
6. Train using that config:

```bash
python -m src.training.train --data data/raw/your_file.csv --config config/detected_dataset_config.json
```

Optional arguments:

- `--output-dir artifacts`
- `--processed-dir data/processed`
- `--top-k-features 20`
- `--test-size 0.15`
- `--validation-size 0.15`

## Run Predictions

```bash
python -m src.inference.predict --input data/processed/sample_predictions_input.csv
```

Prediction output contains:

- original input columns
- `fraud_probability`
- `prediction`
- `prediction_reason`

## Suggested Workflow

1. Download the Kaggle dataset.
2. Inspect columns in `notebooks/eda.ipynb`.
3. Update the target column if needed.
4. Train the models from CLI or notebook.
5. Review figures in `artifacts/figures/`.
6. Use the generated outputs in your report and slides.

## Notes

- The project defaults to binary classification.
- If your dataset has multiple fraud labels, convert them into `0` and `1`.
- The code uses `class_weight="balanced"` before introducing more advanced resampling.
- Tests use synthetic data so the repository remains runnable even before you download the real dataset.
