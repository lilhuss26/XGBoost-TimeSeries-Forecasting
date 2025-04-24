import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def time_series_forecaster(data_path, date_col, target_col, test_size=0.2, forecast_horizon=None):
    """
    Generalized time series forecasting using XGBoost
    
    Parameters:
    - data_path: path to CSV file
    - date_col: name of datetime column
    - target_col: name of target variable column
    - test_size: proportion of data to use for testing (0-1)
    - forecast_horizon: number of future periods to predict (optional)
    """
    
    # Load and prepare data
    df = pd.read_csv(data_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.sort_index(inplace=True)
    
    # Initial visualization
    plt.figure(figsize=(15, 5))
    df[target_col].plot(style='.', title=f'{target_col} Time Series')
    plt.show()
    
    # Create features from datetime index
    def create_features(df):
        df = df.copy()
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofyear'] = df.index.dayofyear
        df['dayofmonth'] = df.index.day
        df['weekofyear'] = df.index.isocalendar().week
        print("features created")
        return df
    
    df = create_features(df)
    
    # Train-test split (can be date-based or random)
    if isinstance(test_size, float):
        # Random split
        train, test = train_test_split(df, test_size=test_size, shuffle=False)
        print("Data splitted")
    else:
        # Date-based split
        train = df.loc[df.index < test_size]
        test = df.loc[df.index >= test_size]
        print("Data splitted")
    
    # Visualize split
    fig, ax = plt.subplots(figsize=(15, 5))
    train[target_col].plot(ax=ax, label='Training Set')
    test[target_col].plot(ax=ax, label='Test Set')
    if isinstance(test_size, str):
        ax.axvline(pd.to_datetime(test_size), color='black', ls='--')
    ax.legend()
    ax.set_title('Train-Test Split')
    plt.show()
    
    # Feature selection
    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
    TARGET = target_col
    
    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]
    print("Data prepared")
    
    # Model training
    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:squarederror',
                           max_depth=3,
                           learning_rate=0.01)
    print("Model init")
    
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)
    print("Model trained")
    # Feature importance
    fi = pd.DataFrame(data=reg.feature_importances_,
                     index=reg.feature_names_in_,
                     columns=['importance'])
    fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
    plt.show()
    
    # Predictions
    test['prediction'] = reg.predict(X_test)
    df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
    
    # Visualization
    ax = df[[target_col]].plot(figsize=(15, 5))
    df['prediction'].plot(ax=ax, style='.')
    plt.legend(['Actual Data', 'Predictions'])
    ax.set_title('Actual vs Predicted')
    plt.show()
    
    # Evaluation
    score = np.sqrt(mean_squared_error(test[target_col], test['prediction']))
    print(f'RMSE Score on Test set: {score:0.2f}')
    
    # Future forecasting (if requested)
    if forecast_horizon:
        last_date = df.index.max()
        future_dates = pd.date_range(start=last_date, periods=forecast_horizon+1, freq='H')[1:]
        future_df = pd.DataFrame(index=future_dates)
        future_df = create_features(future_df)
        future_df['prediction'] = reg.predict(future_df[FEATURES])
        
        # Plot future forecast
        ax = df[[target_col]].plot(figsize=(15, 5))
        future_df['prediction'].plot(ax=ax, style='-')
        plt.legend(['Actual Data', 'Future Predictions'])
        ax.set_title(f'Forecast for next {forecast_horizon} periods')
        plt.show()
        
        return df, future_df
    
    return df

# Example usage:
# df, future = time_series_forecaster('data.csv', 'date', 'sales', forecast_horizon=30)