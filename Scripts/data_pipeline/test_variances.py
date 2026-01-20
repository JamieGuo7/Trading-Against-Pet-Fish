import pandas as pd
import datetime as datetime
import yfinance as yf
import numpy as np

stocks = ["CEG", "KEYS", "LRCX", "HWM", "CARR", "CCK", "EME", "BKNG", "ES", 
          "FBIN", "NVDA", "GOOG", "APD", "CNH", "MCD", "TT", "MRK", "CBOE", "KDP", "EA", 
          "MCK", "AKAM", "NOC", "AMGN", "SBUX", "PHM", "CRWD", "CAT", "HUBB", "WELL", "BAC", 
          "NDAQ", "AAPL",]


results = []

for stock in stocks:
    data = yf.download(stock, start="2025-03-16", end="2025-12-16")
    data["daily_return"] = data["Close"].pct_change()
    returns = data["daily_return"].dropna()

    daily_variance = returns.var()
    monthly_variance = daily_variance*21
    volatility = np.sqrt(monthly_variance)

    results.append({
        "Stock": stock,
        "Monthly Variance": monthly_variance,
        "Volatility": volatility
    })

    df = pd.DataFrame(results)

def get_variances():
    return df.set_index("Stock")["Monthly Variance"]

if __name__ == "__main__":
    print(df)



