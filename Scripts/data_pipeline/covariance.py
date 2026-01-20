import pandas as pd
import numpy as np
import yfinance as yf

stocks = ["CEG", "KEYS", "LRCX", "HWM", "CARR", "CCK", "EME", "BKNG", "ES", 
          "FBIN", "NVDA", "GOOG", "APD", "CNH", "MCD", "TT", "MRK", "CBOE", "KDP", "EA", 
          "MCK", "AKAM", "NOC", "AMGN", "SBUX", "PHM", "CRWD", "CAT", "HUBB", "WELL", "BAC", 
          "NDAQ", "AAPL",]

start_date = "2025-03-16"
end_date = "2025-12-16"
trading_days = 21

prices = yf.download(
    tickers=stocks,
    start=start_date,
    end=end_date,
    auto_adjust=False,
    progress=True
)["Close"]

prices = prices.dropna(axis=1, how="all")

daily_returns = prices.pct_change(fill_method=None)

daily_returns = daily_returns.dropna(how="all")

daily_returns = daily_returns.dropna(axis=1, how="any")

if daily_returns.shape[1] < 2:
    raise RuntimeError("Need at least 2 valid stocks")

daily_cov = daily_returns.cov()

monthly_cov = daily_cov * trading_days

monthly_cov.index.name = "Stock_1"
monthly_cov.columns.name = "Stock_2"

pairs = (
    monthly_cov
    .stack()
    .rename("Monthly_Covariance")
    .reset_index()
)

pairs = pairs[pairs["Stock_1"] <= pairs["Stock_2"]].reset_index(drop=True)
print(pairs.head)

def get_covariance_matrix(): 
    return monthly_cov

monthly_cov.to_csv("monthly_covariance_matrix.csv")
