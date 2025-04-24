import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def time_series_forecaster(data_path, target_col, date_cols=None, test_size=0.2, forecast_horizon=None):
    """
    Generalized time series forecasting using XGBoost
    
    Parameters:
    - data_path: path to CSV file
    - target_col: name of target variable column
    - date_cols: list of columns that make up the date information (e.g., ['year', 'month'])
                 OR name of datetime column (e.g., 'date')
    - test_size: proportion of data to use for testing (0-1) or cutoff date
    - forecast_horizon: number of future periods to predict (optional)
    """
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Handle date columns
    if date_cols is None:
        raise ValueError("Either date_col or date_cols must be specified")
    
    if isinstance(date_cols, str):
        # Single datetime column case
        df[date_cols] = pd.to_datetime(df[date_cols])
        df = df.set_index(date_cols)
        df.sort_index(inplace=True)
        
        # Create features from datetime index
        def create_features(df):
            df = df.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                return df
            df['hour'] = df.index.hour
            df['dayofweek'] = df.index.dayofweek
            df['quarter'] = df.index.quarter
            df['month'] = df.index.month
            df['year'] = df.index.year
            df['dayofyear'] = df.index.dayofyear
            df['dayofmonth'] = df.index.day
            df['weekofyear'] = df.index.isocalendar().week
            return df
        
        has_datetime_index = True
    else:
        # Multiple columns case (year, month, etc.)
        # Create a dummy index (we'll use the date columns directly as features)
        df = df.set_index(pd.RangeIndex(start=0, stop=len(df)))
        has_datetime_index = False
        
        # For multiple date columns, we'll use them directly as features
        def create_features(df):
            return df.copy()
    
    # Initial visualization if we have a datetime index
    if has_datetime_index:
        plt.figure(figsize=(15, 5))
        df[target_col].plot(style='.', title=f'{target_col} Time Series')
        plt.show()
    
    # Create features
    df = create_features(df)
    
    # Convert categorical columns to numeric codes
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col != target_col:  # Don't encode the target variable
            df[col] = pd.factorize(df[col])[0]
    
    # Train-test split
    if isinstance(test_size, float):
        # Random split
        train, test = train_test_split(df, test_size=test_size, shuffle=False)
    else:
        if has_datetime_index:
            # Date-based split
            train = df.loc[df.index < pd.to_datetime(test_size)]
            test = df.loc[df.index >= pd.to_datetime(test_size)]
        else:
            # For non-datetime data, use a simple fraction split
            split_idx = int(len(df) * (1 - test_size))
            train = df.iloc[:split_idx]
            test = df.iloc[split_idx:]
    
    # Visualize split if we have a datetime index
    if has_datetime_index:
        fig, ax = plt.subplots(figsize=(15, 5))
        train[target_col].plot(ax=ax, label='Training Set')
        test[target_col].plot(ax=ax, label='Test Set')
        if isinstance(test_size, str):
            ax.axvline(pd.to_datetime(test_size), color='black', ls='--')
        ax.legend()
        ax.set_title('Train-Test Split')
        plt.show()
    
    # Feature selection
    if has_datetime_index:
        FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
    else:
        # Use the provided date columns plus any other features
        FEATURES = date_cols.copy() if isinstance(date_cols, list) else [date_cols]
    
    # Add any non-date, non-target columns as features
    other_cols = [col for col in df.columns 
                  if col not in [target_col] and col not in FEATURES]
    FEATURES.extend(other_cols)
    
    TARGET = target_col
    
    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]
    
    # Model training
    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                         n_estimators=1000,
                         early_stopping_rounds=50,
                         objective='reg:squarederror',
                         max_depth=3,
                         learning_rate=0.01,
                         enable_categorical=True)  # Enable categorical support
    
    reg.fit(X_train, y_train,
           eval_set=[(X_train, y_train), (X_test, y_test)],
           verbose=100)
    
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
    if has_datetime_index:
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
        if not has_datetime_index:
            print("Warning: Future forecasting requires datetime index. Skipping...")
            return df, None
        
        last_date = df.index.max()
        future_dates = pd.date_range(start=last_date, periods=forecast_horizon+1, freq='D')[1:]
        future_df = pd.DataFrame(index=future_dates)
        future_df = create_features(future_df)
        
        # Ensure future_df has all categorical columns encoded the same way
        for col in categorical_cols:
            if col in FEATURES and col != target_col:
                # Use the same encoding as original data
                future_df[col] = 0  # Default value, adjust as needed
        
        future_df['prediction'] = reg.predict(future_df[FEATURES])
        
        # Plot future forecast
        ax = df[[target_col]].plot(figsize=(15, 5))
        future_df['prediction'].plot(ax=ax, style='-')
        plt.legend(['Actual Data', 'Future Predictions'])
        ax.set_title(f'Forecast for next {forecast_horizon} periods')
        plt.show()
        
        return df, future_df
    
    return df