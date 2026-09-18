"""Download the M5 competition files through Kaggle's official API client."""

from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub

COMPETITION = "m5-forecasting-accuracy"
EXPECTED_FILES = {
    "calendar.csv", "sales_train_evaluation.csv", "sales_train_validation.csv",
    "sample_submission.csv", "sell_prices.csv",
}


def download(output_dir: Path, force: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output_dir.glob("*.csv")}
    if EXPECTED_FILES.issubset(existing) and not force:
        print(f"M5 files already exist in {output_dir.resolve()}")
        return output_dir

    downloaded_path = Path(kagglehub.competition_download(
        COMPETITION, output_dir=str(output_dir), force_download=force
    ))
    search_dir = downloaded_path if downloaded_path.is_dir() else output_dir
    present = {path.name for path in search_dir.rglob("*.csv")}
    missing = sorted(EXPECTED_FILES - present)
    if missing:
        raise FileNotFoundError("Download completed, but files are missing: " + ", ".join(missing))
    print(f"M5 data ready in {search_dir.resolve()}")
    return search_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(args.output_dir, args.force)


if __name__ == "__main__":
    main()
