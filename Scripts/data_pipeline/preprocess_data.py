import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Getting data
tech = ['GOOG', 'MSFT', 'AAPL', 'NVDA', 'AMZN']
data = yf.download(tech, period = '1y', interval='1wk')

# Cleaning and Structuring
data.dropna(inplace = True)
tidy_data = (
    data
    .stack()     # stack tickers
    .stack()     # stack fields
    .reset_index()
)

tidy_data.columns = ["Date", "Ticker", "Type", "Price"]

# Sort nicely
tidy_data = tidy_data.sort_values(["Date", "Ticker", "Type"]).reset_index(drop=True)


print(tidy_data.info())
print(tidy_data.head(10))

file_path = './data/tech_data.csv'
tidy_data.to_csv(file_path, index = False)