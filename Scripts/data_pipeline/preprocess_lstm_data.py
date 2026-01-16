import yfinance as yf
import pandas as pd
import ta

# 1. Getting data
tickers = []
with open('../../data/ESGU_Tickers.txt', 'r') as file:
    for line in file:
        tickers.append(line.strip())

raw_data = yf.download(tickers, period='5d', interval='1d', auto_adjust=True)

print(f"\n Processing {len(tickers)} tickers through full pipeline...")
print("-" * 70)

long_format_data = []

for i, ticker in enumerate(tickers, 1):
    print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")

    print("   - Creating long format data...")
    ticker_df = raw_data.xs(ticker, level='Ticker', axis=1) if len(tickers) > 1 else raw_data
    ticker_df = ticker_df.dropna().drop_duplicates().copy()

    # Calculate indicators again for CSV
    ticker_df['MACD'] = ta.trend.macd(ticker_df['Close'])
    ticker_df['RSI'] = ta.momentum.rsi(ticker_df['Close'])
    ticker_df['STOCH'] = ta.momentum.stoch(ticker_df['High'], ticker_df['Low'],
                                                 ticker_df['Close'])
    ticker_df['CCI'] = ta.trend.cci(ticker_df['High'], ticker_df['Low'],
                                          ticker_df['Close'])
    ticker_df['ADX'] = ta.trend.adx(ticker_df['High'], ticker_df['Low'],
                                          ticker_df['Close'])
    ticker_df['ROC'] = ta.momentum.roc(ticker_df['Close'])
    ticker_df['WILLR'] = ta.momentum.williams_r(ticker_df['High'], ticker_df['Low'],
                                                      ticker_df['Close'])
    ticker_df['ATR'] = ta.volatility.average_true_range(ticker_df['High'], ticker_df['Low'],
                                                              ticker_df['Close'])
    ticker_df['NATR'] = (ticker_df['ATR'] / ticker_df['Close']) * 100
    ticker_df['Ticker'] = ticker
    ticker_df = ticker_df.dropna()

    long_format_data.append(ticker_df)

print(f"\n{'=' * 70}")
print("COMBINING LONG FORMAT DATA...")
print(f"{'=' * 70}")

# Combine all tickers into one DataFrame
long_format = pd.concat(long_format_data, ignore_index=False)
long_format = long_format.reset_index()
long_format = long_format.rename(columns={'index': 'Date'})

# Reorder columns for readability
column_order = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume',
                'MACD', 'RSI', 'STOCH', 'CCI', 'ADX', 'ROC', 'WILLR',
                'ATR', 'NATR']
long_format = long_format[column_order]

# Sort by date and ticker
long_format = long_format.sort_values(['Date', 'Ticker']).reset_index(drop=True)


print(f"   Shape: {long_format.shape}")
print(f"   Rows: {len(long_format):,}")
print(f"   Columns: {len(long_format.columns)}")
print(f"   Date range: {long_format['Date'].min()} to {long_format['Date'].max()}")


LSTM_data_file_path = '../../data/ESGU_LSTM.csv'

long_format.to_csv(LSTM_data_file_path, index=False)