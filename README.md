# Retail Demand Forecasting and Inventory Decisions

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://eric-retail-demand-forecasting.streamlit.app/)
[![Run Complete Project in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmavigo/retail-demand-forecasting/blob/main/notebooks/00_RUN_COMPLETE_PROJECT_IN_COLAB.ipynb)

Portfolio project based on the **M5 Forecasting - Accuracy** dataset. The project turns daily retail history into a reproducible forecasting workflow and a practical inventory decision tool.

Every notebook includes saved outputs from a successful end-to-end run. Recruiters can review the tables, charts, model metrics and conclusions directly on GitHub or Colab without a Kaggle account. Kaggle authentication is required only when someone chooses to rerun the raw-data pipeline, and each person must use their own private token.

## Business question

How many units should each store expect to sell during the next 28 days, and how can that forecast reduce stockouts and excess inventory?

## Why this project matters

The M5 data contains more than five years of daily sales for 30,490 item-store series, including prices, events and calendar variables. It supports a realistic portfolio case with time-series validation, feature engineering, model comparison and business metrics.

## Delivered workflow

1. Data-quality and demand-pattern audit.
2. Seasonal-naive baseline.
3. Global gradient-boosting model using lag, rolling and calendar features.
4. Leakage-free backtesting over a 28-day future horizon.
5. Evaluation with MAE, WAPE and RMSSE.
6. Inventory simulation translating forecast error into stockouts, holding cost and service level.
7. Interactive dashboard for stores, categories and products.

## Current results

The dataset audit passed: 1,941 historical sales days, 30,490 item-store series, 10 stores and 3,049 products. The key calendar and price tables contain no duplicate keys, and `sell_price` has no missing values.

The first 28-day backtest uses `d_1`–`d_1913` for training and `d_1914`–`d_1941` for testing. A 28-day moving average is the strongest initial baseline:

| Model | MAE | WAPE | RMSSE | Bias |
|---|---:|---:|---:|---:|
| Mean of last 28 days | 1.0657 | 73.86% | 0.9240 | 3.91% |
| Seasonal lag 7 | 1.2440 | 86.22% | 1.2010 | 7.36% |
| Seasonal lag 28 | 1.2840 | 89.00% | 1.2445 | 3.91% |
| Last observed value | 1.3730 | 95.16% | 1.2063 | -13.19% |

The best transparent hybrid combines 75% of the moving-average baseline with 25% of the global LightGBM forecast. It reaches **73.77% WAPE**, slightly improving the 73.86% baseline while reducing estimated stockout units by 8.4%.

## Live portfolio dashboard

The repository includes `streamlit_app.py`, a recruiter-friendly presentation of executive KPIs, five-year seasonality, stores, products, model performance and inventory decisions. It uses only derived metrics; the competition CSV files remain local and are excluded from Git.

Run it locally with:

```powershell
streamlit run streamlit_app.py
```

## Dataset setup

The competition files are governed by Kaggle's competition rules and are intentionally excluded from this repository.

First accept the competition rules at:

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Create an API token at `https://www.kaggle.com/settings/api`; keep that secret outside the repository. Every notebook downloads the official competition files directly:

```python
import kagglehub

path = kagglehub.competition_download("m5-forecasting-accuracy")
print("Path to competition files:", path)
```

In Google Colab, add the token to the **Secrets** panel with the name `KAGGLE_API_TOKEN` and enable notebook access. If that secret is missing, every notebook detects the authentication error, opens `kagglehub.login()` and retries the download after you paste the token.

KaggleHub provides these source files:

- `calendar.csv`
- `sales_train_evaluation.csv`
- `sales_train_validation.csv`
- `sample_submission.csv`
- `sell_prices.csv`

## Quick start

Open the complete notebook in Colab, authenticate with Kaggle when prompted and select **Runtime → Run all**. The notebook contains the download, validation, joins, analysis, forecasting and inventory calculations in visible cells.

## Repository structure

```text
notebooks/       Self-contained, editable portfolio analysis
app_data/        Small aggregated tables used by the public dashboard
streamlit_app.py Recruiter-facing interactive presentation
```

The complete work is organized as executable notebooks:

1. **[`00_RUN_COMPLETE_PROJECT_IN_COLAB.ipynb`](notebooks/00_RUN_COMPLETE_PROJECT_IN_COLAB.ipynb)** — easiest option; configures Colab and runs the complete project.
2. [`01_m5_demand_forecasting.ipynb`](notebooks/01_m5_demand_forecasting.ipynb) — data access and statistical baseline.
3. [`02_data_cleaning_and_eda.ipynb`](notebooks/02_data_cleaning_and_eda.ipynb) — cleaning, executive analysis and five-year seasonality.
4. [`03_lightgbm_forecasting.ipynb`](notebooks/03_lightgbm_forecasting.ipynb) — global model, hybrids and holdout evaluation.
5. [`04_inventory_decisions.ipynb`](notebooks/04_inventory_decisions.ipynb) — safety stock, reorder points and inventory trade-offs.

Each notebook independently downloads Kaggle data and performs its own transformations. No hidden project scripts are required. GitHub renders every notebook; JupyterLab, VS Code, Kaggle and Google Colab can run and edit them.

## Portfolio narrative

This project combines data science with operations experience: forecasts are evaluated both statistically and through inventory outcomes. The result shows how model quality changes purchasing decisions, product availability and working capital.
