from time_series_forecaster import time_series_forecaster
import pandas as pd
data = pd.read_csv('data\sales.csv')
#dataDate = pd.read_excel('data\Budget.xlsx')
future = time_series_forecaster(dataframe=data,date_cols=['orderdate_month','orderdate_day','orderdate_year'],target_col='profit',forecast_horizon=12)
#future = time_series_forecaster(dataframe=dataDate,date_cols='EOMonth',target_col='Value',forecast_horizon=12)
print(future)