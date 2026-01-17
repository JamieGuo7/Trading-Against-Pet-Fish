import yfinance as yf
import pandas as pd
from datetime import datetime

# Getting data
tickers = []
with open('./data/ESGU_Tickers.txt', 'r') as file:
    for line in file:
        tickers.append(line.strip())


# Downloading and tidying data
raw_data = yf.download(tickers,
                           period = '3y',
                           interval='1wk',
                           progress = True,
                           auto_adjust = True)

tidy_data = (
        raw_data
        .stack()     # stack tickers
        .stack()     # stack fields
        .reset_index()
    )

tidy_data.columns = ["Date", "Ticker", "Type", "Price"]


tidy_data = tidy_data[tidy_data['Type'] == 'Close'].drop('Type', axis=1)
tidy_data['Date'] = pd.to_datetime(tidy_data['Date'])

# Resampling to ensure each Ticker has exactly one price per week.
tidy_data = (
    tidy_data
    .set_index('Date')
    .groupby('Ticker') # resampling only happens within each ticker's price history
    .resample('W') # Takes the very last price recorded in that week
    .last()
    .drop(columns='Ticker')
    .reset_index()
)

tidy_data = tidy_data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

tidy_data_file_path = '../../data/ESGU_data.csv'
tidy_data.to_csv(tidy_data_file_path, index = False)