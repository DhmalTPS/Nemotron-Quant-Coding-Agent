import json
import numpy as np
from sklearn.decomposition import PCA

# Load input
with open('/input/environment/data/params.json', 'r') as f:
    params = json.load(f)

tickers = params['tickers']
monthly_returns = params['monthly_returns']
n_factors = params['n_factors']
target_weights = params['target_weights']

# Build return matrix R (T x N)
T = len(monthly_returns[tickers[0]])
N = len(tickers)
R = np.zeros((T, N))
for i, ticker in enumerate(tickers):
    R[:, i] = monthly_returns[ticker]

# PCA on returns
pca = PCA(n_components=n_factors)
pca.fit(R)
loadings = pca.components_  # (n_factors, N)
explained_variance = pca.explained_variance_ratio_  # (n_factors,)

# Target weights vector
w = np.array([target_weights[ticker] for ticker in tickers])

# Factor exposures of target portfolio
exp_target = loadings @ w  # (n_factors,)

# Hedge weights via least squares
A = np.vstack([loadings, np.ones((1, N))])  # (n_factors+1, N)
b = np.append(-exp_target, [0])             # (n_factors+1,)
h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # (N,)
w_neutral = w + h

# Neutral factor exposures
exp_neutral = loadings @ w_neutral  # (n_factors,)

# Backtest
factor_scores = pca.transform(R)  # (T, n_factors)
port_ret_neutral = R @ w_neutral    # (T,)
port_ret_target = R @ w             # (T,)

# Sharpe ratios (annualized from monthly)
sharpe_neutral = np.mean(port_ret_neutral) / np.std(port_ret_neutral, ddof=1) * np.sqrt(12)
sharpe_target = np.mean(port_ret_target) / np.std(port_ret_target, ddof=1) * np.sqrt(12)

# Prepare output dictionary
output = {
    "pca_explained_variance_ratio": [round(v, 4) for v in explained_variance],
    "factor_loadings": {
        f"PC{i+1}": {ticker: round(loadings[i, j], 6) for j, ticker in enumerate(tickers)}
        for i in range(n_factors)
    },
    "target_factor_exposures": {
        f"PC{i+1}": round(exp_target[i], 6) for i in range(n_factors)
    },
    "hedge_weights": {ticker: round(h[j], 6) for j, ticker in enumerate(tickers)},
    "neutral_weights": {ticker: round(w_neutral[j], 6) for j, ticker in enumerate(tickers)},
    "neutral_factor_exposures": {
        f"PC{i+1}": round(exp_neutral[i], 6) for i in range(n_factors)
    },
    "portfolio_sharpe_neutral": round(sharpe_neutral, 4),
    "portfolio_sharpe_target": round(sharpe_target, 4)
}

# Write output
write_json("results.json", output)

# Self-checks
sum_neutral_weights = sum(output["neutral_weights"].values())
sum_hedge_weights = sum(output["hedge_weights"].values())
sum_explained_variance = sum(output["pca_explained_variance_ratio"])
max_exp_neutral = max(abs(v) for v in output["neutral_factor_exposures"].values())

if abs(sum_neutral_weights - 1.0) > 1e-5:
    print(f"SELFCHECK_FAIL: sum of neutral_weights is {sum_neutral_weights}, expected ~1.0")
if abs(sum_hedge_weights) > 1e-5:
    print(f"SELFCHECK_FAIL: sum of hedge_weights is {sum_hedge_weights}, expected ~0.0")
if sum_explained_variance > 1.0 + 1e-5:
    print(f"SELFCHECK_FAIL: sum of explained_variance_ratio is {sum_explained_variance}, expected <=1.0")
if max_exp_neutral > 1e-5:
    print(f"SELFCHECK_FAIL: max absolute neutral factor exposure is {max_exp_neutral}, expected ~0.0")

print("Results written to results.json")
print(f"Explained variance ratio: {output['pca_explained_variance_ratio']}")
print(f"Neutral Sharpe: {output['portfolio_sharpe_neutral']}")
print(f"Target Sharpe: {output['portfolio_sharpe_target']}")

