import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from stock_features.add_features import add_features

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Getting data
data_file_path = '../../data/ESGU_data.csv'

data = pd.read_csv(data_file_path)
data_pivot = data.pivot(index='Date', columns='Ticker', values='Price')

returns = data_pivot.pct_change().dropna()
returns = add_features(returns)


def create_4w_target(df):
    df = df.copy()
    df['ret_4w_target'] = df.groupby('Ticker')['ret_1w'].shift(-4)
    return df.dropna(subset=['ret_4w_target'])  # Drop rows without future data

returns = create_4w_target(returns)

train_start_date = '2023-06-01'
train_end_date = '2025-06-01'
val_end_date = '2025-09-01'

train = returns[(train_start_date <= returns['Date']) & (returns['Date'] < train_end_date)]
val = returns[(returns['Date'] >= train_end_date) & (returns['Date'] < val_end_date)]
test = returns[returns['Date'] >= val_end_date]

# We want to maximise predicted returns over one month
target = ['ret_4w_target']
features = ['ret_1w', 'ret_4w', 'momentum_4w', 'roc_12w',  'vol_4w', 'price_ma_ratio']

X_train, y_train = train[features], train[target].squeeze()
X_val,   y_val   = val[features],   val[target].squeeze()
X_test,  y_test  = test[features],  test[target].squeeze()

rf = RandomForestRegressor(
    n_estimators = 1000,
    max_depth = 10,
    min_samples_leaf = 15,
    min_samples_split = 20,
    max_features = 'sqrt',
    bootstrap = True,
    oob_score = True,
    n_jobs = -1,
    random_state = 42,
)

rf.fit(X_train, y_train)

# Validation performance
y_val_pred = rf.predict(X_val)
val_rmse = mean_squared_error(y_val, y_val_pred)
val_r2   = r2_score(y_val, y_val_pred)
print(f"Val RMSE: {val_rmse:.4f}")
print(f"Val R2:   {val_r2:.4f}")
print(f"OOB Score: {rf.oob_score_:.4f}")

# Test performance
y_test_pred = rf.predict(X_test)
test_rmse = mean_squared_error(y_test, y_test_pred)
test_r2   = r2_score(y_test, y_test_pred)
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test R2:   {test_r2:.4f}")

# Save 4 week forward return prediction as predicted return column
test.loc[:, 'predicted_return'] = y_test_pred

portfolio_ranking = test.groupby('Date').apply(
    lambda x: x[['Date', 'Ticker', 'predicted_return']].nlargest(10, 'predicted_return')
).reset_index(drop=True)

print("\nTop 10 stocks by predicted return (4 weeks forward!):")
latest_date = test['Date'].max()
latest_date_ranking = portfolio_ranking[portfolio_ranking['Date'] == latest_date]
print(latest_date_ranking)