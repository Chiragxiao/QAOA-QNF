import yfinance as yf
import pandas as pd

tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'JNJ', 'JPM', 'XOM', 'GLD']
start_date = '2019-01-01'
end_date = '2024-01-01'

data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']

print(data.head())
data.to_csv('portfolio_prices.csv')
