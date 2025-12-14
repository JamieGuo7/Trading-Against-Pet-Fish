import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Getting data
tickers = []

with open('data/ESGU_Tickers.txt', 'r') as file:
    for line in file:
        tickers.append(line.strip())


# Downloading and tidying data
raw_data = yf.download(tickers,
                           period = '1y',
                           interval='1wk',
                           progress = True,
                           auto_adjust = True) # Fix Warning
raw_data.dropna(inplace = True)

tidy_data = (
        raw_data
        .stack()     # stack tickers
        .stack()     # stack fields
        .reset_index()
    )

tidy_data.columns = ["Date", "Ticker", "Type", "Price"]
tidy_data = tidy_data.sort_values(["Date", "Ticker", "Type"]).reset_index(drop=True)
tidy_data = tidy_data[tidy_data['Type'] == 'Close']

file_path = './data/ESGU_data.csv'
tidy_data.to_csv(file_path, index = False)