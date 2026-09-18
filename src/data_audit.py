"""Create a lightweight, reproducible audit of the M5 competition files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_FILES = {
    "calendar.csv",
    "sales_train_evaluation.csv",
    "sales_train_validation.csv",
    "sample_submission.csv",
    "sell_prices.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def csv_shape(path: Path) -> tuple[int, int, list[str]]:
    header = pd.read_csv(path, nrows=0)
    rows = sum(1 for _ in path.open("rb")) - 1
    return rows, len(header.columns), header.columns.tolist()


def build_audit(data_dir: Path) -> dict:
    present = {path.name for path in data_dir.glob("*.csv")}
    missing = sorted(EXPECTED_FILES - present)
    if missing:
        raise FileNotFoundError(
            "Missing M5 files in " + str(data_dir.resolve()) + ": " + ", ".join(missing)
        )

    files: dict[str, dict] = {}
    for name in sorted(EXPECTED_FILES):
        path = data_dir / name
        rows, columns, column_names = csv_shape(path)
        files[name] = {
            "size_mb": round(path.stat().st_size / 1_048_576, 2),
            "rows": rows,
            "columns": columns,
            "first_columns": column_names[:12],
        }

    calendar = pd.read_csv(data_dir / "calendar.csv")
    prices = pd.read_csv(data_dir / "sell_prices.csv")
    sales_header = pd.read_csv(data_dir / "sales_train_evaluation.csv", nrows=0)
    day_columns = [column for column in sales_header.columns if column.startswith("d_")]

    return {
        "dataset": "M5 Forecasting - Accuracy",
        "files": files,
        "time_coverage": {
            "start": str(pd.to_datetime(calendar["date"]).min().date()),
            "end": str(pd.to_datetime(calendar["date"]).max().date()),
            "calendar_days": int(len(calendar)),
            "sales_days": len(day_columns),
        },
        "dimensions": {
            "stores": int(prices["store_id"].nunique()),
            "items_with_prices": int(prices["item_id"].nunique()),
            "store_item_price_records": int(len(prices)),
            "events": int(calendar["event_name_1"].nunique(dropna=True)),
        },
        "quality_checks": {
            "duplicate_calendar_days": int(calendar["d"].duplicated().sum()),
            "duplicate_price_keys": int(
                prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).sum()
            ),
            "missing_prices": int(prices["sell_price"].isna().sum()),
        },
    }


def to_markdown(audit: dict) -> str:
    coverage = audit["time_coverage"]
    dimensions = audit["dimensions"]
    quality = audit["quality_checks"]
    file_rows = "\n".join(
        f"| {name} | {meta['rows']:,} | {meta['columns']:,} | {meta['size_mb']:.2f} |"
        for name, meta in audit["files"].items()
    )
    return f"""# M5 data audit

## Coverage

- Calendar: {coverage['start']} to {coverage['end']}
- Calendar days: {coverage['calendar_days']:,}
- Historical sales days: {coverage['sales_days']:,}
- Stores: {dimensions['stores']:,}
- Items with price records: {dimensions['items_with_prices']:,}
- Events: {dimensions['events']:,}

## Files

| File | Rows | Columns | MB |
|---|---:|---:|---:|
{file_rows}

## Quality checks

- Duplicate calendar keys: {quality['duplicate_calendar_days']:,}
- Duplicate store-item-week price keys: {quality['duplicate_price_keys']:,}
- Missing prices: {quality['missing_prices']:,}
"""


def main() -> None:
    args = parse_args()
    audit = build_audit(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (args.output_dir / "data_audit.md").write_text(to_markdown(audit), encoding="utf-8")
    print(to_markdown(audit))


if __name__ == "__main__":
    main()

