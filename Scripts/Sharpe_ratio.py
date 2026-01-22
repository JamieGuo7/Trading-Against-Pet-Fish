import numpy as np
import pandas as pd
from scipy.optimize import minimize

COV_PATH = "./results/market_covariance_matrix.csv"
FC_PATH  = "./results/latest_forecasts.csv"

# -----------------------
# 1) Load + align inputs
# -----------------------
cov = pd.read_csv(COV_PATH, index_col=0)
fc  = pd.read_csv(FC_PATH)

cov.columns = cov.columns.astype(str)
cov.index   = cov.index.astype(str)

required_cols = {"ticker", "forecast_return"}
missing = required_cols - set(fc.columns)
if missing:
    raise ValueError(f"latest_forecasts.csv is missing columns: {missing}")

fc["ticker"] = fc["ticker"].astype(str)

common = [t for t in cov.columns if t in set(fc["ticker"])]
if len(common) < 2:
    raise ValueError(f"Need at least 2 overlapping tickers. Found {len(common)}.")

Sigma = cov.loc[common, common].to_numpy(dtype=float)

mu = (
    fc.set_index("ticker")
      .loc[common, "forecast_return"]
      .astype(float)
      .to_numpy()
)

# If forecast_return is in percent (e.g. 2.3 means 2.3%), convert to decimals:
mu = mu / 100.0

# -----------------------
# 2) Stabilize covariance
# -----------------------
Sigma = 0.5 * (Sigma + Sigma.T)

# Stronger ridge helps a lot with optimizer stability
ridge = 1e-4  # <- increased from 1e-6
Sigma = Sigma + ridge * np.eye(Sigma.shape[0])

# -----------------------
# 3) Helpers
# -----------------------
def portfolio_stats(w: np.ndarray):
    """Return (expected_return, volatility, sharpe) with rf=0."""
    r = float(w @ mu)
    v2 = float(w @ Sigma @ w)
    v = np.sqrt(max(v2, 0.0))
    s = r / v if v > 0 else -1e12
    return r, v, s

n = len(common)
x0 = np.ones(n) / n
bounds = [(0.0, 1.0)] * n
constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

# -----------------------
# 4) Robust long-only optimizer:
#    mean-variance utility, then pick best Sharpe
# -----------------------
def solve_mean_variance(lam: float):
    """
    Maximize: mu^T w - 0.5*lam*w^T Sigma w
    s.t. w>=0, sum(w)=1
    """
    def objective(w):
        return -(mu @ w - 0.5 * lam * (w @ Sigma @ w))

    def grad(w):
        # gradient of objective (negative of utility gradient)
        return -(mu - lam * (Sigma @ w))

    res = minimize(
        objective,
        x0,
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 5000, "ftol": 1e-12, "disp": False},
    )
    return res

# Sweep lambdas (risk aversion). Wider range = more likely to find good Sharpe.
lams = np.logspace(-2, 4, 25)  # 0.01 ... 10000
best = None

for lam in lams:
    res = solve_mean_variance(lam)
    if not res.success:
        continue
    w = np.clip(res.x, 0.0, 1.0)
    w = w / w.sum()
    r, v, s = portfolio_stats(w)
    if (best is None) or (s > best["sharpe"]):
        best = {"lam": lam, "w": w, "ret": r, "vol": v, "sharpe": s}

if best is None:
    raise RuntimeError("All optimizations failed. Try increasing ridge further (e.g. 1e-3) or check data quality.")

# -----------------------
# 5) Report results
# -----------------------
w = best["w"]
weights = pd.Series(w, index=common).sort_values(ascending=False)

print("\nBest long-only portfolio found (by Sharpe) from mean-variance sweep:")
print(f"Chosen lambda:    {best['lam']:.6g}")
print(f"Expected return:  {best['ret']:.6f}")
print(f"Volatility:       {best['vol']:.6f}")
print(f"Sharpe (rf=0):    {best['sharpe']:.4f}")

print("\nTop 15 weights (LONG-ONLY):")
print(weights.head(15).to_string())

out_path = "./optimal_long_only_weights.csv"
weights.rename("weight").to_csv(out_path, header=True)
print(f"\nSaved weights to: {out_path}")
