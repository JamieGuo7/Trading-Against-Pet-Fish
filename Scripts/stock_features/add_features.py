import pandas as pd

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_features(returns):
    feature_dfs = []

    for ticker in returns.columns:
        ticker_df = returns[[ticker]].copy()
        ticker_df.columns = ['ret_1w']
        ticker_df['Ticker'] = ticker

        ticker_df['ret_4w'] = ticker_df['ret_1w'].rolling(4).sum()
        ticker_df['vol_4w'] = ticker_df['ret_1w'].rolling(4).std()

        prices = (1 + ticker_df['ret_1w']).cumprod()
        ticker_df['price_ma_ratio'] = prices / prices.rolling(4).mean()

        ticker_df['rsi'] = calculate_rsi(prices)

        ticker_df['momentum_4w'] = prices / prices.shift(4) - 1
        ticker_df['roc_12w'] = prices.pct_change(12)

        feature_dfs.append(ticker_df.dropna())

    result = pd.concat(feature_dfs, axis=0).reset_index()

    column_order = [
        'Date', 'Ticker',  # identifiers
        'ret_1w',
        'ret_4w', 'momentum_4w', 'roc_12w',
        'vol_4w',
        'price_ma_ratio', 'rsi'
    ]

    return result[column_order]