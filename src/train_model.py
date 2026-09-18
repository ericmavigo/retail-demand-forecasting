"""Train a global LightGBM model and create inventory decision outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HORIZON = 28
TRAIN_END = 1913


def score(actual: np.ndarray, predicted: np.ndarray, train: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    scale = np.mean(np.diff(train, axis=1) ** 2, axis=1)
    usable = scale > 0
    denominator = np.abs(actual).sum()
    return {
        "MAE": float(np.abs(error).mean()),
        "WAPE": float(np.abs(error).sum() / denominator),
        "RMSSE": float(np.sqrt(np.mean(error[usable] ** 2, axis=1) / scale[usable]).mean()),
        "Bias": float(error.sum() / denominator),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--samples", type=int, default=800_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    sales = pd.read_csv(args.data_dir / "sales_train_evaluation.csv")
    calendar = pd.read_csv(args.data_dir / "calendar.csv", parse_dates=["date"])
    day_cols = [column for column in sales if column.startswith("d_")]
    calendar = calendar.set_index("d").loc[day_cols].reset_index()
    values = sales[day_cols].to_numpy(dtype=np.float32)
    train = values[:, :TRAIN_END]
    actual = values[:, TRAIN_END:TRAIN_END + HORIZON]

    category_columns = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    category_codes = {
        column: pd.Categorical(sales[column]).codes.astype(np.int16) for column in category_columns
    }
    rng = np.random.default_rng(42)
    series_index = rng.integers(0, len(sales), size=args.samples)
    day_index = rng.integers(365, TRAIN_END, size=args.samples)

    def sampled_features(sidx: np.ndarray, didx: np.ndarray, history: np.ndarray) -> pd.DataFrame:
        features = pd.DataFrame({
            "lag_1": history[sidx, didx - 1],
            "lag_7": history[sidx, didx - 7],
            "lag_14": history[sidx, didx - 14],
            "lag_28": history[sidx, didx - 28],
            "lag_56": history[sidx, didx - 56],
            "month": calendar.loc[didx, "month"].to_numpy(dtype=np.int8),
            "weekday": calendar.loc[didx, "wday"].to_numpy(dtype=np.int8),
            "event": calendar.loc[didx, "event_name_1"].notna().to_numpy(dtype=np.int8),
        })
        features["rolling_7"] = np.array([history[s, d-7:d].mean() for s, d in zip(sidx, didx)], dtype=np.float32)
        features["rolling_28"] = np.array([history[s, d-28:d].mean() for s, d in zip(sidx, didx)], dtype=np.float32)
        for column in category_columns:
            features[column] = category_codes[column][sidx]
        return features

    X = sampled_features(series_index, day_index, train)
    y = train[series_index, day_index]
    model = lgb.LGBMRegressor(
        objective="poisson", n_estimators=450, learning_rate=0.05, num_leaves=64,
        max_depth=-1, min_child_samples=100, subsample=0.8, colsample_bytree=0.9,
        reg_lambda=0.2, random_state=42, n_jobs=-1, verbosity=-1,
    )
    model.fit(X, y, categorical_feature=category_columns)

    history = np.concatenate([train, np.zeros((len(sales), HORIZON), dtype=np.float32)], axis=1)
    forecast = np.zeros_like(actual)
    all_series = np.arange(len(sales))
    for step in range(HORIZON):
        day = TRAIN_END + step
        X_future = sampled_features(all_series, np.full(len(sales), day), history)
        forecast[:, step] = np.maximum(0, model.predict(X_future)).astype(np.float32)
        history[:, day] = forecast[:, step]

    baseline = np.repeat(train[:, -HORIZON:].mean(axis=1, keepdims=True), HORIZON, axis=1)
    predictions = {
        "mean_last_28": baseline,
        "hybrid_25pct_lgbm": 0.75 * baseline + 0.25 * forecast,
        "hybrid_50pct_lgbm": 0.50 * baseline + 0.50 * forecast,
        "hybrid_75pct_lgbm": 0.25 * baseline + 0.75 * forecast,
        "global_lightgbm": forecast,
    }
    metrics = pd.DataFrame([
        {"model": name, **score(actual, prediction, train)}
        for name, prediction in predictions.items()
    ]).sort_values("WAPE")
    best_name = metrics.iloc[0]["model"]
    best_forecast = predictions[best_name]
    metrics.to_csv(args.reports_dir / "model_metrics.csv", index=False)

    forecast_daily = pd.DataFrame({
        "date": calendar.loc[TRAIN_END:TRAIN_END + HORIZON - 1, "date"].to_numpy(),
        "actual_units": actual.sum(axis=0),
        "baseline_units": baseline.sum(axis=0),
        "model_units": best_forecast.sum(axis=0),
    })
    forecast_daily.to_csv(args.output_dir / "forecast_daily.csv", index=False)

    store_rows = []
    for store in sorted(sales["store_id"].unique()):
        mask = sales["store_id"].eq(store).to_numpy()
        store_rows.append({
            "store_id": store,
            "actual_units": float(actual[mask].sum()),
            "forecast_units": float(best_forecast[mask].sum()),
            "WAPE": float(np.abs(actual[mask] - best_forecast[mask]).sum() / actual[mask].sum()),
        })
    pd.DataFrame(store_rows).to_csv(args.output_dir / "forecast_by_store.csv", index=False)

    importance = pd.DataFrame({
        "feature": model.feature_name_, "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    importance.to_csv(args.output_dir / "feature_importance.csv", index=False)

    residual_std = (train[:, -84:] - np.repeat(train[:, -112:-84].mean(axis=1, keepdims=True), 84, axis=1)).std(axis=1)
    safety_stock = 1.65 * residual_std * np.sqrt(7)
    model_order = best_forecast.sum(axis=1)
    baseline_order = baseline.sum(axis=1)
    actual_total = actual.sum(axis=1)
    inventory = pd.DataFrame({
        "id": sales["id"], "item_id": sales["item_id"], "store_id": sales["store_id"],
        "actual_units": actual_total, "model_order_units": model_order,
        "baseline_order_units": baseline_order, "safety_stock": safety_stock,
        "reorder_point": best_forecast[:, :7].sum(axis=1) + safety_stock,
        "model_stockout_units": np.maximum(actual_total - model_order, 0),
        "model_excess_units": np.maximum(model_order - actual_total, 0),
        "baseline_stockout_units": np.maximum(actual_total - baseline_order, 0),
        "baseline_excess_units": np.maximum(baseline_order - actual_total, 0),
    })
    inventory.to_parquet(args.output_dir / "inventory_decisions.parquet", index=False)
    inventory_summary = pd.DataFrame({
        "method": ["Baseline", best_name],
        "stockout_units": [inventory["baseline_stockout_units"].sum(), inventory["model_stockout_units"].sum()],
        "excess_units": [inventory["baseline_excess_units"].sum(), inventory["model_excess_units"].sum()],
    })
    inventory_summary.to_csv(args.output_dir / "inventory_summary.csv", index=False)
    Path(args.reports_dir / "model_summary.json").write_text(json.dumps({
        "best_model": metrics.iloc[0]["model"], "best_wape": float(metrics.iloc[0]["WAPE"]),
        "training_samples": args.samples,
    }, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(inventory_summary.to_string(index=False))


if __name__ == "__main__":
    main()
