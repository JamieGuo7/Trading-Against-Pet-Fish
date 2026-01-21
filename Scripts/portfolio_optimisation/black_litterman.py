import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import black_litterman, BlackLittermanModel, EfficientFrontier

# Getting data
STOCKS = []
with open('../../data/ESGU_Tickers.txt', 'r') as file:
    for line in file:
        STOCKS.append(line.strip())

# UNCOMMENT for our selection of stocks
# STOCKS = ["CEG", "KEYS", "LRCX", "HWM", "CARR", "CCK", "EME", "BKNG", "ES",
#           "FBIN", "NVDA", "GOOG", "APD", "CNH", "MCD", "TT", "MRK", "CBOE", "KDP", "EA",
#           "MCK", "AKAM", "NOC", "AMGN", "SBUX", "PHM", "CRWD", "CAT", "HUBB", "WELL", "BAC",
#           "NDAQ", "AAPL"]

# Data Cleaning
results_df = pd.read_csv('../../results/results.csv')
results_df = results_df.drop_duplicates(subset='ticker')
results_df = results_df[results_df['ticker'].isin(STOCKS)].set_index('ticker')
results_df = results_df.dropna(subset=['forecast_return', 'test_rmse'])

# Re-index to match the order of our stocks list exactly
results_df = results_df.reindex(STOCKS)


# Load Covariance and dropping duplicates
cov_df = pd.read_csv('../../data/market_covariance_matrix.csv', index_col=0)
cov_df = cov_df.loc[~cov_df.index.duplicated(), ~cov_df.columns.duplicated()]

# Only keep stocks for which we have covariances
STOCKS = results_df.index.intersection(cov_df.index).tolist()

# Align indices
S = cov_df.loc[STOCKS, STOCKS]
results_df = results_df.loc[STOCKS]

# Extract Views (Q) and RMSEs for Confidence (Omega)
lstm_views = (results_df['forecast_return'] / 100).to_dict()
rmses = results_df['test_rmse'].values

print("Fetching market caps...")
tickers_obj = yf.Tickers(" ".join(STOCKS))
# Convert market caps to a Series aligned with our Covariance Matrix index
mcaps = pd.Series({t: tickers_obj.tickers[t].info.get('marketCap', 1e9) for t in STOCKS})


pi = black_litterman.market_implied_prior_returns(
    mcaps,
    risk_aversion=2.5,
    cov_matrix=S
)

# We use the squared RMSE to tell the model how much to trust each individual stock's forecast.
tau = 0.025
P = np.eye(len(STOCKS))
view_variances = np.diag(S)
confidence_multiplier = (rmses / rmses.mean())**2
omega = np.diag(tau * view_variances * confidence_multiplier)

# Diagnostics to evaluate model performance
print("\n--- Black-Litterman Diagnostics ---")
print(f"Tau: {tau}")
print(f"\nOmega diagonal values (view uncertainties):")
print(f"  Min: {np.min(np.diag(omega)):.6f}")
print(f"  Max: {np.max(np.diag(omega)):.6f}")
print(f"  Mean: {np.mean(np.diag(omega)):.6f}")

print(f"\nPrior covariance (tau * S) diagonal:")
print(f"  Mean: {np.mean(np.diag(tau * S)):.6f}")

ratio = np.mean(np.diag(omega)) / np.mean(np.diag(tau * S))
print(f"\nRatio of view uncertainty to prior uncertainty: {ratio:.2f}")
print("  (Should be roughly 0.1 to 10 for balanced influence)")

print(f"\nRMSE Statistics:")
print(f"  Min: {rmses.min():.4f}")
print(f"  Max: {rmses.max():.4f}")
print(f"  Mean: {rmses.mean():.4f}")

print(f"\nMonthly Volatility Statistics:")
monthly_vols = np.sqrt(np.diag(S))
print(f"  Min: {monthly_vols.min():.4f}")
print(f"  Max: {monthly_vols.max():.4f}")
print(f"  Mean: {monthly_vols.mean():.4f}")

# Check RMSE vs Volatility
print(f"\nRMSE/Volatility Ratios (forecast accuracy check):")
rmse_to_vol = rmses / monthly_vols
print(f"  Min: {rmse_to_vol.min():.2f}")
print(f"  Max: {rmse_to_vol.max():.2f}")
print(f"  Mean: {rmse_to_vol.mean():.2f}")
print("  (Ratios < 1.0 suggest predictions better than random)")

# Show which stocks have best/worst predictions
print(f"\nTop 5 Most Confident Predictions (lowest RMSE):")
top_confident = results_df.nsmallest(5, 'test_rmse')[['forecast_return', 'test_rmse']]
for ticker, row in top_confident.iterrows():
    print(f"  {ticker}: Forecast={row['forecast_return']:.2f}%, RMSE={row['test_rmse']:.4f}")

print(f"\nTop 5 Least Confident Predictions (highest RMSE):")
least_confident = results_df.nlargest(5, 'test_rmse')[['forecast_return', 'test_rmse']]
for ticker, row in least_confident.iterrows():
    print(f"  {ticker}: Forecast={row['forecast_return']:.2f}%, RMSE={row['test_rmse']:.4f}")

bl = BlackLittermanModel(
    S,
    pi=pi,
    absolute_views=lstm_views,
    omega=omega,
    tau=tau
)

# Combined Returns and Covariance
ret_bl = bl.bl_returns()
S_bl = bl.bl_cov()

print("\n--- Black-Litterman Posterior Returns ---")
print(f"Mean posterior return: {ret_bl.mean():.4f}")
print(f"Min posterior return: {ret_bl.min():.4f}")
print(f"Max posterior return: {ret_bl.max():.4f}")

# Optimisation
ef = EfficientFrontier(ret_bl, S_bl)
ef.add_constraint(lambda w: w >= 0) # No short selling

try:
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    print("\n--- Optimized Portfolio Weights ---")
    sorted_weights = sorted(cleaned_weights.items(), key=lambda x: x[1], reverse=True)
    for ticker, weight in sorted_weights:
        if weight > 0:
            print(f"{ticker}: {weight:.2%}")

    # Portfolio performance
    perf = ef.portfolio_performance(verbose=False)
    print(f"\n--- Expected Portfolio Performance ---")
    print(f"Expected Monthly Return: {perf[0]:.2%}")
    print(f"Monthly Volatility: {perf[1]:.2%}")
    print(f"Sharpe Ratio: {perf[2]:.3f}")

except Exception as e:
    print(f"Optimization failed: {e}. Defaulting to Min Volatility.")
    weights = ef.min_volatility()
    print(ef.clean_weights())