# XGBoost Time Series Forecasting

A powerful and flexible time series forecasting tool using XGBoost, designed to handle various temporal patterns and provide accurate predictions.

## Features

- **Advanced Time Series Features**
  - Temporal features (year, month, day, dayofweek, etc.)
  - Lag features with multiple windows
  - Rolling statistics (mean, std, min, max, median)
  - Exponential weighted moving averages
  - Seasonal features using sine/cosine transformations
  - Feature interactions

- **Model Capabilities**
  - XGBoost-based forecasting
  - Automatic hyperparameter tuning
  - Multiple evaluation metrics (R², RMSE, MAE)
  - Feature importance visualization
  - Future forecasting support

- **Data Handling**
  - Automatic datetime index handling
  - Categorical feature encoding
  - Missing value handling
  - Train-test splitting (random or date-based)

## Installation

```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn
```

## Usage

```python
from time_series_forecaster import time_series_forecaster

# Example usage
future = time_series_forecaster(
    dataframe=data,
    target_col='Value',
    date_cols=['orderdate_month', 'orderdate_day', 'orderdate_year'],
    forecast_horizon=12
)
```

### Parameters

- `dataframe`: Input DataFrame containing the time series data
- `target_col`: Name of the target column to forecast
- `date_cols`: List of date columns or single date column name
- `test_size`: Size of the test set (float or date string)
- `forecast_horizon`: Number of periods to forecast (optional)

### Output

The function returns:
- Model predictions for the test set
- Future forecasts (if forecast_horizon is specified)
- Visualizations of:
  - Time series data
  - Feature importance
  - Model fit (R²)
  - Actual vs Predicted values

## Model Evaluation

The model provides multiple evaluation metrics:
- R² Score (coefficient of determination)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

## Example

```python
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')

# Run the forecaster
results = time_series_forecaster(
    dataframe=data,
    target_col='sales',
    date_cols='date',
    test_size=0.2,
    forecast_horizon=30
)
```

## Visualization

The tool automatically generates several visualizations:
1. Time Series Plot
2. Feature Importance Plot
3. R² Visualization
4. Actual vs Predicted Plot

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- xgboost
- scikit-learn

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

