#!/usr/bin/env python3
"""Train and evaluate a logistic regression model for UFC bout winners."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from ufc_pred_model.model import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_model,
)

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1]
    / "data/large_clean.csv"
)
def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = set(FEATURES + [TARGET])
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    data = data[data[TARGET].isin(["Red", "Blue"])].copy()
    if data.empty:
        raise ValueError(f"{path} contains no Red/Blue winner rows")
    return data

def train_and_evaluate(data: pd.DataFrame, test_size: float = 0.2) -> tuple[Pipeline, dict[str, float]]:
    """Fit on the earlier rows and evaluate on the later rows."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    holdout_size = int(len(data) * test_size)

    if holdout_size < 2 or holdout_size >= len(data) - 1:
        raise ValueError("dataset is too small for the requested time split")

    train, test = data.iloc[holdout_size:], data.iloc[:holdout_size]
    model = build_model()
    model.fit(train[FEATURES], train[TARGET])
    predictions = model.predict(test[FEATURES])
    probabilities = model.predict_proba(test[FEATURES])[:, list(model.classes_).index("Red")]
    metrics = {
        "accuracy": accuracy_score(test[TARGET], predictions),
        "balanced_accuracy": balanced_accuracy_score(test[TARGET], predictions),
        "roc_auc_red": roc_auc_score((test[TARGET] == "Red").astype(int), probabilities),
    }

    print(f"Rows: {len(data)} (train={len(train)}, test={len(test)})")
    print("Time split: older training rows, newest holdout rows; no shuffling")
    print("Metrics:")

    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print("\nClassification report:")
    print(classification_report(test[TARGET], predictions, zero_division=0))
    return model, metrics


def predict_bout(model: Pipeline, bout: dict[str, Any]) -> dict[str, Any]:
    """Predict a bout from feature values using the same interface as training."""
    features = pd.DataFrame([bout]).reindex(columns=FEATURES)
    probabilities = model.predict_proba(features)[0]
    predicted = model.classes_[probabilities.argmax()]
    return {
        "winner": str(predicted),
        "probability_red": float(
            probabilities[list(model.classes_).index("Red")]
        ),
        "probability_blue": float(
            probabilities[list(model.classes_).index("Blue")]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--predict-json",
        type=Path,
        help="JSON object containing matchup features; prints one prediction",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    data = load_data(args.dataset)
    model, _ = train_and_evaluate(data, args.test_size)
    if args.predict_json:
        with args.predict_json.open(encoding="utf-8") as file:
            bout = json.load(file)
        print("\nPrediction:")
        print(json.dumps(predict_bout(model, bout), indent=2))


if __name__ == "__main__":
    main()