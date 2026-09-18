# Baseline backtest

Forecast horizon: 28 days (`d_1914`–`d_1941`).

| Model | MAE | WAPE | RMSSE | Bias |
|---|---:|---:|---:|---:|
| mean_last_28 | 1.0657 | 73.86% | 0.9240 | 3.91% |
| seasonal_7 | 1.2440 | 86.22% | 1.2010 | 7.36% |
| seasonal_28 | 1.2840 | 89.00% | 1.2445 | 3.91% |
| last_value | 1.3730 | 95.16% | 1.2063 | -13.19% |
