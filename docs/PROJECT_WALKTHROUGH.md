# Retail Demand Forecasting: Project Walkthrough

This guide explains the complete analysis behind the [published Streamlit dashboard](https://eric-retail-demand-forecasting.streamlit.app/) and the reproducible [Colab notebooks](https://github.com/ericmavigo/retail-demand-forecasting/tree/main/notebooks). It follows the decisions in the order they were made: define the operating question, understand the source tables, check data quality, build a fair time-series test, compare models, and translate forecast error into an inventory illustration.

## Executive answer

The project asks how much demand each product-store combination may have over the next 28 days, and whether that estimate is useful for planning replenishment.

There is no single winner at every level of aggregation. For 30,490 individual item-store series, the most reliable result is a transparent blend of the 28-day mean baseline and global LightGBM: 75% baseline plus 25% LightGBM. Its WAPE is 73.77%, a very small improvement over the baseline's 73.86%. Pure LightGBM is worse on this item-store test (76.54% WAPE), so it should not replace the simple benchmark at this level.

For the separate comparison of the 10 aggregated store-total series, global LightGBM is the clear winner at 7.85% WAPE. That result answers a different forecasting question and must not be compared directly with SKU-by-store error. The practical conclusion is to select models for the decision grain: keep the simple or lightly blended forecast for detailed replenishment experiments, and use the store-level LightGBM result as evidence that shared machine learning can help when demand is aggregated.

The inventory simulation makes the forecast trade-off concrete. Under its stated assumptions, the 25% LightGBM blend reduces estimated stockout units by 8.4% versus the 28-day baseline, while increasing excess units. The M5 files do not contain actual on-hand inventory, supplier lead times, purchase orders, lost sales or carrying costs. These are scenario results, not measured historical savings or a production reorder recommendation.

## Step 1 — Define the question and the decision grain

The target is daily unit demand, not revenue or profit. The fine-grained series is one product in one store. There are 3,049 products and 10 stores, giving 30,490 item-store series. The operating horizon is 28 days because the M5 competition asks for that future window and because it provides a concrete planning interval.

We also create a second view by summing all products and categories within each store. This produces 10 store-total series. It allows classical time-series approaches to be compared without fitting a separate complex statistical model to each of 30,490 sparse series. Aggregation smooths intermittent item demand, so it changes the problem and the expected error scale.

## Step 2 — Identify the source tables and their relationships

The project uses the official M5 competition files:

| File | What it contains | Role in the analysis |
|---|---|---|
| `sales_train_evaluation.csv` | Product/store hierarchy and daily unit sales in columns `d_1` through `d_1941` | Demand history and the final observed 28-day test window |
| `sales_train_validation.csv` | The same hierarchy with sales through `d_1913` | Confirms the competition's original training cutoff |
| `calendar.csv` | Maps day IDs to dates, retail weeks, weekdays, events and SNAP flags | Date alignment and future-known calendar features |
| `sell_prices.csv` | Weekly prices keyed by store, product and retail week | Revenue estimates and price coverage audit |
| `sample_submission.csv` | Series IDs and 28 forecast placeholders | Output schema only; placeholder zeros are not observed demand |

Sales is stored as a wide matrix: each row is a product-store series and each day is a column. A day key such as `d_1941` joins to `calendar.d`. The calendar's retail-week key joins to `sell_prices` on the composite key `(store_id, item_id, wm_yr_wk)`. This is why price is weekly even though demand is daily.

The data catalog notebook classifies fields before analysis. IDs and hierarchy labels are treated as keys or categories, daily sales as nonnegative counts, prices as continuous values, calendar fields as temporal/categorical fields, and event/SNAP fields as flags. This prevents applying numeric summaries to IDs or interpreting blank event labels as failed records.

## Step 3 — Audit quality before calculating results

The audit checks file presence, dimensions, column names, types, date continuity, unique keys, hierarchy coverage, domain validity, missingness and cross-table joins. The audit recorded 30,490 evaluation series, 1,941 observed days, 3,049 products and 10 stores. The wide sales matrix has 1,947 columns: six hierarchy identifiers plus 1,941 day columns.

Key checks passed: the calendar day and date keys are unique; the weekly product-price composite key is unique; series IDs are unique; validation and evaluation row order agrees; and evaluation adds exactly 28 days beyond validation. The price table has 6,841,121 records across 282 retail weeks, no missing prices, and every positive-sales observation has a joined weekly price. Blank event fields are expected: they mean that no named event is listed for that date.

About 68% of item-store-day observations are zero. Those zeros are valid demand observations in this dataset, not missing values. This matters for both model choice and evaluation: item-level demand is intermittent, and an error of one unit can be large relative to a low-volume series. The source has no stock-on-hand field, so a recorded zero means zero sales, not necessarily proven zero demand caused by availability.

The complete quality checks and table previews are in [00 · Data catalog and quality](https://colab.research.google.com/github/ericmavigo/retail-demand-forecasting/blob/main/notebooks/00_DATA_CATALOG_AND_QUALITY.ipynb).

## Step 4 — Join and summarize without creating an unnecessary giant table

The sales data has roughly 59 million item-store-day values when expanded to long form. The notebooks keep the main sales matrix wide for forecasting and aggregate across its daily columns when creating overall time trends. Calendar and price joins are inspected on small, explicit previews, and the product-price relation is validated with its composite business key before it is used.

Revenue in the dashboard is an analytical estimate: daily units multiplied by the applicable weekly selling price. It is not audited net sales because M5 does not provide transaction IDs, returns, discounts at transaction time, tax or customer baskets. For the same reason, an average basket size or true average ticket cannot be derived.

## Step 5 — Explore demand before selecting a model

The exploratory analysis aggregates demand over time and across the hierarchy to inspect five-year trends, week-of-year seasonality, weekday patterns, monthly averages, store/category/product contribution and event-day associations. It also examines variation by store so that an apparently good total forecast does not hide a location with substantially different errors.

The M5 calendar contains national, religious, cultural and sporting labels. Event-day lifts in the dashboard are descriptive comparisons with normal weekdays; they are not causal effects. Event labels overlap with season, weekday and other factors, and event dates were not randomized.

These views are developed in [02 · Data cleaning and EDA](https://colab.research.google.com/github/ericmavigo/retail-demand-forecasting/blob/main/notebooks/02_data_cleaning_and_eda.ipynb) and summarized in the dashboard's Seasonality and Stores & products sections.

## Step 6 — Design a test that respects time

Random train/test splitting would mix later observations into the training data and give an unrealistic answer to a forecasting question. Instead, the test is fixed at the final 28 observed days:

- Training history: `d_1`–`d_1913`.
- Held-out actuals: `d_1914`–`d_1941`.
- Forecast horizon: 28 consecutive days.

Every candidate for a given comparison is scored against the same held-out dates and the same set of series. The test actuals are used only to calculate metrics after each forecast has been made. In the recursive LightGBM forecast, days 2–28 use earlier model predictions for their lag features; actual held-out sales are never fed back into later test predictions.

This is one historical holdout, not a rolling-origin benchmark over many seasons. It is honest for the stated split but does not establish stability across all years, promotions or future operating regimes.

## Step 7 — Establish simple baselines first

The project compares four useful baseline ideas: carry the last value forward, repeat the average of the last 7 or 28 days, repeat the last observed 7-day pattern, and repeat the last 28-day pattern. These baselines are easy to understand, fast to reproduce and difficult for a complex model to beat without a real signal.

At the item-store level, the 28-day moving average is the strongest initial benchmark. It is less sensitive to a single unusual day than the last-value method and less tied to the exact calendar alignment than copying the last four weeks day by day.

| Item-store model | MAE (units/day) | WAPE | RMSSE | Bias |
|---|---:|---:|---:|---:|
| Mean of last 28 days | 1.0657 | 73.86% | 0.9240 | +3.91% |
| Seasonal lag 7 | 1.2440 | 86.22% | 1.2010 | +7.36% |
| Seasonal lag 28 | 1.2840 | 89.00% | 1.2445 | +3.91% |
| Last observed value | 1.3730 | 95.16% | 1.2063 | −13.19% |

The exact metrics and score definitions are in [01 · Statistical baselines](https://colab.research.google.com/github/ericmavigo/retail-demand-forecasting/blob/main/notebooks/01_m5_demand_forecasting.ipynb).

## Step 8 — Add global LightGBM and explain the feature design

LightGBM is a gradient-boosted decision-tree model. One global model is trained across many item-store-day examples instead of fitting an unrelated model for each series. The model can share patterns across products and stores while using the IDs and hierarchy as categorical inputs. A Poisson objective is used because the target is nonnegative count demand.

The feature set contains information available when each forecast is made:

- Demand lags at 1, 7, 14, 28 and 56 days capture recent level and weekly/recent history.
- Rolling means over 7 and 28 prior days smooth daily volatility.
- Weekday and month represent calendar seasonality.
- An event indicator represents whether the known calendar date has a named event.
- Product, department, category, store and state codes give the global model hierarchical context.

Training rows are sampled reproducibly from historical days after enough lag history exists. Each feature row is built only from days strictly before its target day. This avoids label leakage. At prediction time, the model forecasts one day at a time, appends that prediction to a temporary history, and uses those predicted values to construct later lags for the 28-day path.

LightGBM is included because it can learn nonlinear interactions among product/store categories, recent demand and known calendar features while sharing statistical strength across a large panel. It is not selected merely because it is more complex: its 76.54% item-store WAPE is worse than the simple baseline, so its pure forecast is rejected at that grain.

The comparison tests transparent blends: 75% baseline + 25% LightGBM, 50/50, and 25% baseline + 75% LightGBM. Blending is a controlled way to ask whether the machine-learning signal adds value without allowing a noisier model to replace the stronger benchmark outright.

| Item-store candidate | MAE | WAPE | RMSSE | Bias |
|---|---:|---:|---:|---:|
| 75% baseline + 25% LightGBM | 1.0643 | **73.77%** | 0.9195 | +1.88% |
| Mean of last 28 days | 1.0657 | 73.86% | 0.9240 | +3.91% |
| 50% baseline + 50% LightGBM | 1.0706 | 74.20% | 0.9210 | −0.15% |
| 25% baseline + 75% LightGBM | 1.0841 | 75.14% | 0.9281 | −2.17% |
| Global LightGBM | 1.1044 | 76.54% | 0.9404 | −4.20% |

The blend reduces WAPE by only about 0.09 percentage points against the baseline. This is a narrow win, not a major model breakthrough. The model notebook deliberately keeps the pure LightGBM and weaker blends visible so the comparison cannot hide the trade-off.

## Step 9 — Compare model families at a fair operational level

Prophet, NeuralProphet and SARIMAX are useful comparators because they express trend, seasonality and known calendar effects differently from lag-based boosted trees. Fitting these individual-series approaches to all 30,490 item-store series would be unnecessarily slow for this portfolio comparison. The notebook therefore aggregates all products into the 10 store-total series and scores every candidate on those same stores and the same final 28 days.

| Store-total model | MAE (units/day) | WAPE | Bias |
|---|---:|---:|---:|
| Global LightGBM | 345.31 | **7.85%** | +5.03% |
| Seasonal naive, 28 days | 401.85 | 9.13% | +3.91% |
| Seasonal naive, 7 days | 427.89 | 9.73% | +7.36% |
| Prophet | 428.65 | 9.74% | +7.28% |
| NeuralProphet | 579.47 | 13.17% | −8.47% |
| SARIMAX | 579.69 | 13.18% | +4.41% |
| Mean of last 28 days | 606.34 | 13.78% | +3.91% |

The choices answer different modeling questions:

| Model family | Why it is included | What this evaluation says |
|---|---|---|
| Seasonal-naive baselines | Establish whether recent level or repeated weekly patterns are already sufficient | Strong reference points; they remain hard to beat for sparse item-store series |
| Global LightGBM | Learn nonlinear effects and share information across product/store series using lags, hierarchy and future-known calendar fields | Best of the store-total models, but weaker than the baseline as a pure item-store predictor |
| Prophet | Interpretable trend plus weekly/yearly seasonal components and named events | Close to the store-level weekly seasonal-naive benchmark, but not the winner here |
| NeuralProphet | Neural autoregression with weekly/yearly seasonality and event inputs | More flexible architecture did not improve this small set of 10 aggregate series in this run |
| SARIMAX | Classical autoregression with a 7-day seasonal term and future-known weekday, annual Fourier and event regressors | A transparent classical comparator; it is materially less accurate than store-level LightGBM here |

For Prophet, NeuralProphet and SARIMAX, the notebook includes weekly/yearly components and event information in a model-appropriate form. Those future calendar fields are known in advance; future sales remain hidden until scoring. The store-level scores above are not comparable with the per-item-store table because the sums, series count and error scale differ.

## Step 10 — Choose metrics that describe different failure modes

- **MAE** is the average absolute error in units per series per day. It is interpretable in the target unit but can be dominated by high-volume series when errors are pooled.
- **WAPE** is total absolute error divided by total actual units. It describes absolute error relative to overall demand, but can obscure performance on low-volume series.
- **RMSSE** scales squared forecast errors by in-sample day-to-day changes, then averages across usable series. It helps compare series with different scales and penalizes large misses more strongly.
- **Bias** is signed aggregate error divided by total actual demand in this notebook's scoring function (`actual − forecast`). Positive bias therefore means aggregate underforecasting; negative bias means overforecasting.

No single score describes every operational outcome. The dashboard presents several metrics, and the inventory notebook asks what the errors imply under explicit replenishment assumptions.

## Step 11 — Translate the forecast into a transparent inventory scenario

For each item-store series, the simulator sums the 28 daily predictions. It estimates the standard deviation of recent forecast residuals, multiplies it by `z = 1.65` and `sqrt(7)` to calculate safety stock for a seven-day lead time, and sets the reorder point to predicted lead-time demand plus safety stock. The `z` value is an approximate 95% service target under simplified distribution assumptions.

The scenario compares the 28-day baseline with the 25% LightGBM blend over the held-out period:

| Forecast policy | Estimated stockout units | Estimated excess units |
|---|---:|---:|
| 28-day baseline | 196,782 | 148,644 |
| 75% baseline + 25% LightGBM | 180,256 | 157,087 |

That is about 16,526 fewer stockout units (8.4% lower) and about 8,443 more excess units (5.7% higher). This makes the trade-off explicit: a forecast can reduce underprediction by ordering more, while shifting some risk to overstock. The notebook also applies a configurable 3-to-1 stockout-to-holding penalty to rank scenarios. That ratio is an illustrative business assumption; it is not a cost measured from the M5 data.

Actual on-hand inventory, replenishment orders, supplier variability, lead times, lost-sales records, shelf-life and item-specific economics would be required to estimate real service levels or financial impact. The simulator is an educational decision framework, not an operational order generator.

## Final conclusion and next steps

The project reaches its conclusion in stages: audit first, inspect temporal and hierarchy structure, protect a future holdout, establish honest baselines, add machine learning, compare all candidates at a stated grain, and finally evaluate a downstream inventory scenario. The evidence supports a nuanced choice rather than a blanket claim that machine learning wins:

1. At SKU-by-store level, retain the 28-day mean as the trusted benchmark. The 25% LightGBM blend narrowly improves the holdout and reduces estimated underforecast units in the simulator, but the accuracy gain is very small and excess inventory increases.
2. At store-total level, global LightGBM is the strongest tested forecast at 7.85% WAPE. This is a better supported use of the shared nonlinear model in these results.
3. Before operational use, repeat the evaluation with rolling-origin cutoffs, more seasonal cycles and business-approved cost/service assumptions. Validate item-level performance by category and demand volume, and incorporate real inventory and replenishment data.

## Reproduce the analysis

The saved notebook outputs let a reader review results without Kaggle credentials. To rerun the full pipeline in Google Colab, accept the M5 competition rules, provide a personal Kaggle token through Colab Secrets as `KAGGLE_API_TOKEN`, and run the notebooks from top to bottom. The token is never committed to this repository.

1. [00 · Data catalog and quality](https://colab.research.google.com/github/ericmavigo/retail-demand-forecasting/blob/main/notebooks/00_DATA_CATALOG_AND_QUALITY.ipynb)
2. [00 · Complete project in Colab](https://colab.research.google.com/github/ericmavigo/retail-demand-forecasting/blob/main/notebooks/00_RUN_COMPLETE_PROJECT_IN_COLAB.ipynb)
3. [01 · Statistical baselines](https://colab.research.google.com/github/ericmavigo/retail-demand-forecasting/blob/main/notebooks/01_m5_demand_forecasting.ipynb)
4. [02 · Data cleaning and EDA](https://colab.research.google.com/github/ericmavigo/retail-demand-forecasting/blob/main/notebooks/02_data_cleaning_and_eda.ipynb)
5. [03 · LightGBM and model comparison](https://colab.research.google.com/github/ericmavigo/retail-demand-forecasting/blob/main/notebooks/03_lightgbm_forecasting.ipynb)
6. [04 · Inventory decisions](https://colab.research.google.com/github/ericmavigo/retail-demand-forecasting/blob/main/notebooks/04_inventory_decisions.ipynb)

The raw competition files are not committed because they are subject to Kaggle's competition terms. The public Streamlit app uses small derived summary tables and does not need a viewer to provide a Kaggle token.

