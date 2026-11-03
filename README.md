# Weekly Sales Forecasting: Corporación Favorita

A time-series forecasting project using Python and XGBoost to predict weekly sales for Corporación Favorita, an Ecuadorian grocery retailer. The project evaluates whether machine learning can improve on simple historical benchmarks when forecasting aggregate weekly sales.

## Project Overview

This project uses the publicly available Corporación Favorita Store Sales dataset on Kaggle to investigate weekly sales patterns and build a one-week-ahead forecasting model.

Daily sales data from January 2013 to August 2017 were aggregated into weekly totals across all stores and product families. An XGBoost regression model was then developed to predict the week-on-week change in sales, using historical sales patterns and calendar features.

The model was evaluated against two baseline forecasts:

- **Previous-week baseline:** assumes next week's sales will equal the most recent week's sales.
- **Previous-year baseline:** uses sales from the equivalent week in the previous year.

## Methodology

### Data preparation

- Aggregated daily sales into weekly totals, with weeks ending on Sunday.
- Removed the partial first and last weeks, and the first 52 observations required for the one-year lag feature.

### Feature engineering

The model used:

- Sales lags of 1, 2, 4 and 52 weeks.
- A rolling four-week average of historical sales.
- Calendar features: week of year and month.

Historical features were shifted so the model could not access information from the week being forecast.

### Model

The project uses `XGBRegressor` with the following fixed parameters:

- 300 estimators
- Learning rate: 0.05
- Maximum tree depth: 4

Rather than predicting the absolute sales level directly, the model predicts the change from the previous week's sales. This predicted change is then added to the previous week's actual sales to produce the forecast.

### Validation

A five-split expanding-window `TimeSeriesSplit` was used to preserve chronological order and evaluate the model across five consecutive test windows.

Performance was assessed using:

- Root Mean Squared Error (RMSE)
- R-squared (R²)

## Results

| Forecasting approach | Overall RMSE |
|---|---|
| XGBoost | 579,602 |
| Previous-week baseline | 600,671 |
| Previous-year baseline | 1,125,041 |

XGBoost achieved approximately 3.5% lower RMSE than the previous-week baseline and approximately 48.5% lower RMSE than the previous-year baseline.

However, the improvement over the previous-week baseline was inconsistent: XGBoost only outperformed it in the fifth validation window. Overall R² was also close to zero, indicating that the model explained little of the variation in weekly sales.

The results therefore suggest that the model offers a modest improvement in aggregate error, but does not establish a consistently reliable forecasting advantage over a simple last-week benchmark.

## Additional Analysis: April 2016 Earthquake

The project also examined the impact of the April 2016 Ecuador earthquake on weekly sales.

The week ending 24 April 2016 recorded approximately 7.48 million in sales, compared with approximately 5.34 million in the preceding week, an increase of around 40%.

A product-family breakdown showed notable increases in groceries, beverages and cleaning products. This pattern is consistent with emergency purchasing or stockpiling, although the analysis does not establish a causal relationship.

A manually defined earthquake-period indicator was also tested as an additional feature. It produced only a negligible change in overall RMSE, suggesting that this simple event flag did not materially improve forecasting performance.

## Business Relevance

Accurate sales forecasts can support retail decisions involving:

- Inventory planning and replenishment.
- Product availability and stock management.
- Purchasing and supplier coordination.
- Workforce and operational planning.

This project demonstrates how historical sales data can be transformed into a forecasting workflow and evaluated against practical benchmarks. It also highlights why a lower overall error does not necessarily mean a model is consistently better across different periods.

## Limitations

- The model forecasts aggregate weekly sales rather than individual stores or product families.
- Only one machine-learning algorithm was evaluated.
- Hyperparameters were fixed rather than systematically tuned.
- External variables such as promotions, oil prices and holidays were not incorporated into the main model.
- The model's improvement over the previous-week baseline was inconsistent.
- No formal residual analysis or financial impact assessment was conducted.

## Future Development

Potential extensions include:

- Incorporating promotional activity, holiday information and oil prices.
- Building separate forecasting models for individual stores or product families.
- Testing explicit trend features and longer rolling averages.
- Comparing additional forecasting algorithms.
- Conducting more detailed error and residual analysis.

## Tools and Technologies

- **Python**
- **pandas**: data preparation and aggregation
- **NumPy**: numerical operations
- **XGBoost**: regression modelling
- **scikit-learn**: time-series validation and evaluation metrics

## Repository Structure

```
weekly-sales-forecasting/
├── benchmark.py
└── README.md
```

The repository contains the modelling script and project documentation. The Kaggle dataset is not included and must be downloaded separately.

## Reproducibility

1. Download the Corporación Favorita dataset from Kaggle and unzip it into a folder called `store-sales-time-series-forecasting` in the project directory.
2. Install the required Python libraries:

   ```
   pip install pandas numpy scikit-learn xgboost
   ```

3. Run the modelling script from the project directory:

   ```
   python benchmark.py
   ```

---

An independent data analytics project exploring time-series forecasting, feature engineering, model evaluation and the practical limitations of aggregate retail sales prediction.
