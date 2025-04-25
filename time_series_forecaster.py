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
        
        # Create features from datetime index - only using the specified date_col
        def create_features(df):
            df = df.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                return df
            # Only create features from the specified date column
            df[date_cols] = df.index
            return df
        
        has_datetime_index = True
    else:
        # Multiple columns case (year, month, etc.)
        # Create a datetime index from the date columns
        date_cols_str = [str(col) for col in date_cols]
        df['date'] = pd.to_datetime(df[date_cols_str].astype(str).agg('-'.join, axis=1))
        df = df.set_index('date')
        df.sort_index(inplace=True)
        
        def create_features(df):
            return df.copy()
        
        has_datetime_index = True
    
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
    
    # Feature selection - use all columns except target
    FEATURES = [col for col in df.columns if col != target_col]
    
    TARGET = target_col
    
    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]
    
    # Model training
    reg = xgb.XGBRegressor(base_score=0.5, booster='gblinear',
                        n_estimators=500,
                        early_stopping_rounds=50,
                        alpha=0.1,          
                        lambda_=1.0,
                        objective='reg:squarederror',
                        #max_depth=3,
                        learning_rate=0.01,
                        enable_categorical=True)  
    
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
        
        # Create future date columns
        if isinstance(date_cols, list):
            for col in date_cols:
                if 'year' in col:
                    future_df[col] = future_df.index.year
                elif 'month' in col:
                    future_df[col] = future_df.index.month
                elif 'day' in col:
                    future_df[col] = future_df.index.day
        
        # Add all required features to future_df
        for feature in FEATURES:
            if feature not in future_df.columns:
                if feature in df.columns:
                    # Use median value from training data for numerical features
                    if pd.api.types.is_numeric_dtype(df[feature]):
                        future_df[feature] = train[feature].median()
                    else:
                        # For categoricals, use the most common value
                        future_df[feature] = train[feature].mode()[0]
                else:
                    future_df[feature] = 0
        
        future_df['prediction'] = reg.predict(future_df[FEATURES])
        
        # Plot future forecast
        ax = df[[target_col]].plot(figsize=(15, 5))
        future_df['prediction'].plot(ax=ax, style='-')
        plt.legend(['Actual Data', 'Future Predictions'])
        ax.set_title(f'Forecast for next {forecast_horizon} periods')
        plt.show()
        
        return df, future_df
    
    return df