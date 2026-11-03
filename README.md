# Weekly Sales Forecasting: Corporación Favorita

## Goal
Forecast weekly total unit sales for the Favorita store network, using time-aware validation and benchmarks against naive forecasts.

## Data
Kaggle "Store Sales - Time Series Forecasting" (train.csv), daily sales by store and product family, 2013-01-01 to 2017-08-15. The data measures sales volume, not revenue. Weeks are Sunday-ending (W-SUN). The partial first and last weeks are dropped, leaving 2013-01-13 to 2017-08-13.

## Features and target
Target: weekly total sales, modelled as the week-on-week change (y minus last week's sales).
Features (past values only, via shift): lags of 1, 2, 4 and 52 weeks, a 4-week rolling mean, week of year and month.

## Validation
TimeSeriesSplit with 5 expanding-window folds. Test windows:
1. 2014-08-31 to 2015-03-29
2. 2015-04-05 to 2015-11-01
3. 2015-11-08 to 2016-06-05
4. 2016-06-12 to 2017-01-08
5. 2017-01-15 to 2017-08-13

Standard shuffled k-fold is not used, because it would train on future data.

## Baselines
- Naive 1: last week's sales repeated.
- Naive 52: sales from the same week last year.

## Results (mean across folds)
| Model | RMSE | R² |
|---|---|---|
| XGBoost (change target) | 579,602 | -0.03 |
| Naive 1 (last week) | 600,671 | -0.33 |
| Naive 52 (last year) | 1,125,041 | -3.29 |

Per fold, XGBoost has lower RMSE than last week's sales in only fold 5. It has lower RMSE than last year's sales in folds 1, 2, 3 and 5.

## Findings
- The mean RMSE advantage over last week's sales comes entirely from fold 5. In folds 1 to 4 the naive forecast is better.
- R² is near zero for XGBoost. It explains almost none of the variation in weekly sales on these test windows.
- Fold 3 is the worst for all models. Its test window contains the week ending 24 April 2016, when weekly sales rose to about 7.5 million against roughly 5.1 to 5.8 million in surrounding weeks.
- Product-family breakdown of that week: about 2.0 million of the roughly 2.1 million extra sales came from groceries, beverages, cleaning, personal care and home care. The earthquake of 16 April 2016 is recorded in holidays_events.csv as a national event. The most likely explanation is relief buying, but the data shows what sold, not why, so this is an inference.

## Limitations
- Five folds with one shock is a small sample; the results are not a general accuracy claim.
- The model has no information about external events, so shocks such as the earthquake are unpredictable and also distort lag and rolling features for the following weeks.
- Hyperparameters were set once and not tuned, to avoid fitting the test folds.

## Next steps
- Run one experiment with an event flag for the earthquake weeks, reported as a separate row. The baseline results above stay unchanged.
