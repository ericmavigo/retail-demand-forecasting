"""Backtest simple 28-day demand forecasts on the M5 evaluation data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HORIZON = 28
TRAIN_END = 1913


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def metrics(actual: np.ndarray, predicted: np.ndarray, train: np.ndarray) -> dict:
    error = actual - predicted
    mae = float(np.abs(error).mean())
    denominator = float(np.abs(actual).sum())
    wape = float(np.abs(error).sum() / denominator) if denominator else float("nan")

    scale = np.mean(np.diff(train, axis=1) ** 2, axis=1)
    mse = np.mean(error**2, axis=1)
    usable = scale > 0
    rmsse = float(np.sqrt(mse[usable] / scale[usable]).mean())
    bias = float(error.sum() / denominator) if denominator else float("nan")
    return {"MAE": mae, "WAPE": wape, "RMSSE": rmsse, "Bias": bias}


def main() -> None:
    args = parse_args()
    source = args.data_dir / "sales_train_evaluation.csv"
    train_columns = [f"d_{day}" for day in range(1, TRAIN_END + 1)]
    test_columns = [f"d_{day}" for day in range(TRAIN_END + 1, TRAIN_END + HORIZON + 1)]
    frame = pd.read_csv(source, usecols=train_columns + test_columns)
    train = frame[train_columns].to_numpy(dtype=np.float32)
    actual = frame[test_columns].to_numpy(dtype=np.float32)

    forecasts = {
        "last_value": np.repeat(train[:, -1:], HORIZON, axis=1),
        "mean_last_28": np.repeat(train[:, -HORIZON:].mean(axis=1, keepdims=True), HORIZON, axis=1),
        "seasonal_7": np.tile(train[:, -7:], (1, HORIZON // 7)),
        "seasonal_28": train[:, -HORIZON:].copy(),
    }

    rows = []
    for model, prediction in forecasts.items():
        rows.append({"model": model, **metrics(actual, prediction, train)})
    result = pd.DataFrame(rows).sort_values("WAPE").reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "baseline_metrics.csv", index=False)
    display = result.copy()
    for column in ["WAPE", "Bias"]:
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    for column in ["MAE", "RMSSE"]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")

    markdown = "# Baseline backtest\n\n"
    markdown += "Forecast horizon: 28 days (`d_1914`–`d_1941`).\n\n"
    markdown += "| Model | MAE | WAPE | RMSSE | Bias |\n"
    markdown += "|---|---:|---:|---:|---:|\n"
    for row in display.itertuples(index=False):
        markdown += f"| {row.model} | {row.MAE} | {row.WAPE} | {row.RMSSE} | {row.Bias} |\n"
    (args.output_dir / "baseline_results.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
