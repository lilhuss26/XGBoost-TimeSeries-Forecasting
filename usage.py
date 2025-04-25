from time_series_forecaster import time_series_forecaster
import pandas as pd
data = pd.read_csv('data\sales.csv')
future = time_series_forecaster(dataframe=data,date_cols=['orderdate_month','orderdate_day','orderdate_year'],target_col='sales',forecast_horizon=12)
print(future)