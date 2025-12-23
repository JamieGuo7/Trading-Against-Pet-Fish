import yfinance as yf
import pandas as pd
from datetime import datetime

# Getting data
tickers = []
with open('../../data/ESGU_Tickers.txt', 'r') as file:
    for line in file:
        tickers.append(line.strip())


# Downloading and tidying data
raw_data = yf.download(tickers,
                           period = '3y',
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
tidy_data = tidy_data[tidy_data['Type'] == 'Close']
tidy_data['Date'] = pd.to_datetime(tidy_data['Date'])
tidy_data = tidy_data.sort_values(["Date", "Ticker", "Type"]).reset_index(drop=True)

train_end_date = '2025-08-30'
validation_end_date = '2025-10-30'
test_end_date = '2025-12-31'

train_data = tidy_data[tidy_data['Date'] <= train_end_date]
validation_data = tidy_data[(tidy_data['Date'] <= validation_end_date)
                            & (tidy_data['Date'] > train_end_date)]
test_data = tidy_data[(tidy_data['Date'] <= test_end_date)
                            & (tidy_data['Date'] > validation_end_date)]

train_data = train_data.sort_values(['Ticker', 'Date'])
validation_data = validation_data.sort_values(['Ticker', 'Date'])
test_data = test_data.sort_values(['Ticker', 'Date'])

train_data_file_path = '../../data/training_data.csv'
validation_data_file_path = '../../data/validation_data.csv'
test_data_file_path = '../../data/test_data.csv'

train_data.to_csv(train_data_file_path, index = False)
validation_data.to_csv(validation_data_file_path, index = False)
test_data.to_csv(test_data_file_path, index = False)