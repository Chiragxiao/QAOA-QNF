import pandas as pd
import numpy as np

# Load prices data
prices = pd.read_csv('portfolio_prices.csv', index_col='Date', parse_dates=True)

# Calculate log returns
returns = np.log(prices / prices.shift(1)).dropna()

# Inspect data
print(returns.head())

split_date = '2023-01-01'
train_returns = returns[returns.index < split_date]
val_returns = returns[returns.index >= split_date]

print(train_returns.shape, val_returns.shape)
