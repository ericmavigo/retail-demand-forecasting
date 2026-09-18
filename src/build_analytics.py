"""Build clean, small analytical tables from the raw M5 files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sales = pd.read_csv(args.data_dir / "sales_train_evaluation.csv")
    calendar = pd.read_csv(args.data_dir / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(args.data_dir / "sell_prices.csv")
    day_cols = [column for column in sales if column.startswith("d_")]
    calendar = calendar.set_index("d").loc[day_cols].reset_index()
    units = sales[day_cols].to_numpy(dtype=np.float32)

    row_index = pd.MultiIndex.from_frame(sales[["store_id", "item_id"]])
    week_values = calendar["wm_yr_wk"].drop_duplicates().tolist()
    price_wide = prices.pivot(index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price")
    price_wide = price_wide.reindex(index=row_index, columns=week_values)
    week_position = {week: position for position, week in enumerate(week_values)}
    day_week_positions = np.array([week_position[week] for week in calendar["wm_yr_wk"]])

    stores = sorted(sales["store_id"].unique())
    categories = sorted(sales["cat_id"].unique())
    departments = sorted(sales["dept_id"].unique())
    states = sorted(sales["state_id"].unique())
    store_map = {value: index for index, value in enumerate(stores)}
    category_map = {value: index for index, value in enumerate(categories)}
    department_map = {value: index for index, value in enumerate(departments)}
    state_map = {value: index for index, value in enumerate(states)}
    n_days = len(day_cols)

    store_units = np.zeros((len(stores), n_days), dtype=np.float64)
    store_revenue = np.zeros_like(store_units)
    category_units = np.zeros((len(categories), n_days), dtype=np.float64)
    category_revenue = np.zeros_like(category_units)
    department_units = np.zeros((len(departments), n_days), dtype=np.float64)
    department_revenue = np.zeros_like(department_units)
    state_units = np.zeros((len(states), n_days), dtype=np.float64)
    state_revenue = np.zeros_like(state_units)
    row_revenue = np.zeros(len(sales), dtype=np.float64)
    missing_price_sales = 0

    for start in range(0, len(sales), 1000):
        stop = min(start + 1000, len(sales))
        block_units = units[start:stop]
        block_prices = price_wide.iloc[start:stop].to_numpy(dtype=np.float32)[:, day_week_positions]
        missing_price_sales += int(((block_units > 0) & np.isnan(block_prices)).sum())
        block_revenue = block_units * np.nan_to_num(block_prices, nan=0.0)
        row_revenue[start:stop] = block_revenue.sum(axis=1)
        meta = sales.iloc[start:stop]
        for label, target_units, target_revenue, mapping in [
            ("store_id", store_units, store_revenue, store_map),
            ("cat_id", category_units, category_revenue, category_map),
            ("dept_id", department_units, department_revenue, department_map),
            ("state_id", state_units, state_revenue, state_map),
        ]:
            codes = meta[label].map(mapping).to_numpy()
            for code in np.unique(codes):
                mask = codes == code
                target_units[code] += block_units[mask].sum(axis=0)
                target_revenue[code] += block_revenue[mask].sum(axis=0)

    def panel(labels: list[str], unit_array: np.ndarray, revenue_array: np.ndarray, key: str) -> pd.DataFrame:
        return pd.concat([
            pd.DataFrame({
                key: label,
                "date": calendar["date"],
                "units": unit_array[index],
                "estimated_revenue": revenue_array[index],
            }) for index, label in enumerate(labels)
        ], ignore_index=True)

    daily = pd.DataFrame({
        "date": calendar["date"],
        "units": units.sum(axis=0),
        "estimated_revenue": store_revenue.sum(axis=0),
        "event_name": calendar["event_name_1"].fillna("None"),
        "event_type": calendar["event_type_1"].fillna("None"),
        "weekday": calendar["weekday"],
        "month": calendar["month"],
        "year": calendar["year"],
    })
    daily["avg_selling_price"] = daily["estimated_revenue"].div(daily["units"]).replace([np.inf], np.nan)
    daily["moving_average_28"] = daily["units"].rolling(28, min_periods=1).mean()
    daily.to_csv(args.output_dir / "daily_overview.csv", index=False)

    store_daily = panel(stores, store_units, store_revenue, "store_id")
    category_daily = panel(categories, category_units, category_revenue, "cat_id")
    department_daily = panel(departments, department_units, department_revenue, "dept_id")
    state_daily = panel(states, state_units, state_revenue, "state_id")
    store_daily.to_parquet(args.output_dir / "store_daily.parquet", index=False)
    category_daily.to_parquet(args.output_dir / "category_daily.parquet", index=False)
    department_daily.to_parquet(args.output_dir / "department_daily.parquet", index=False)
    state_daily.to_parquet(args.output_dir / "state_daily.parquet", index=False)

    weekly = daily.set_index("date")[["units", "estimated_revenue"]].resample("W-SUN").sum().reset_index()
    weekly["year"] = weekly["date"].dt.year
    weekly["week_of_year"] = weekly["date"].dt.isocalendar().week.astype(int)
    weekly.to_csv(args.output_dir / "weekly_overview.csv", index=False)

    def summary(panel_df: pd.DataFrame, key: str) -> pd.DataFrame:
        result = panel_df.groupby(key, as_index=False).agg(
            units=("units", "sum"), estimated_revenue=("estimated_revenue", "sum")
        )
        result["avg_selling_price"] = result["estimated_revenue"] / result["units"]
        return result.sort_values("estimated_revenue", ascending=False)

    summary(store_daily, "store_id").to_csv(args.output_dir / "store_summary.csv", index=False)
    summary(category_daily, "cat_id").to_csv(args.output_dir / "category_summary.csv", index=False)
    summary(department_daily, "dept_id").to_csv(args.output_dir / "department_summary.csv", index=False)

    product = sales[["item_id", "cat_id", "dept_id"]].copy()
    product["units"] = units.sum(axis=1)
    product["estimated_revenue"] = row_revenue
    product = product.groupby(["item_id", "cat_id", "dept_id"], as_index=False).agg(
        units=("units", "sum"), estimated_revenue=("estimated_revenue", "sum")
    ).sort_values("estimated_revenue", ascending=False)
    product["revenue_share"] = product["estimated_revenue"] / product["estimated_revenue"].sum()
    product["cumulative_revenue_share"] = product["revenue_share"].cumsum()
    product["abc_class"] = np.select(
        [product["cumulative_revenue_share"] <= 0.80, product["cumulative_revenue_share"] <= 0.95],
        ["A", "B"], default="C"
    )
    product.to_csv(args.output_dir / "product_summary.csv", index=False)

    event_summary = daily[daily["event_name"] != "None"].groupby(
        ["event_name", "event_type"], as_index=False
    ).agg(days=("date", "count"), avg_units=("units", "mean"), avg_revenue=("estimated_revenue", "mean"))
    normal_by_weekday = daily[daily["event_name"] == "None"].groupby("weekday")["units"].mean()
    event_summary["normal_units"] = event_summary["event_name"].map(
        daily.set_index("event_name")["weekday"].to_dict()
    ).map(normal_by_weekday)
    event_summary["unit_lift"] = event_summary["avg_units"] / event_summary["normal_units"] - 1
    event_summary.to_csv(args.output_dir / "event_summary.csv", index=False)

    validation = {
        "rows": int(len(sales)), "sales_days": n_days, "total_units": int(units.sum()),
        "estimated_revenue": float(store_revenue.sum()),
        "zero_share": float((units == 0).mean()),
        "duplicate_ids": int(sales["id"].duplicated().sum()),
        "negative_sales": int((units < 0).sum()),
        "missing_price_sales": missing_price_sales,
    }
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
