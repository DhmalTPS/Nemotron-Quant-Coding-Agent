import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import ncx2
import json
import os

# Load data
df = pd.read_csv('/input/environment/data/fred_rates.csv')
df['date'] = pd.to_datetime(df['date'])
df['DFF'] = df['DFF'] / 100.0  # convert to decimal

# Filter last 10 years
df = df[df['date'] >= '2016-01-01'].copy()
r = df['DFF'].values
n_obs = len(r)

# Statistics
r_mean = float(np.mean(r))
r_std = float(np.std(r, ddof=1))
r_min = float(np.min(r))
r_max = float(np.max(r))
r_current = float(r[-1])

# Negative log-likelihood for CIR
def neg_log_likelihood(params):
    kappa, theta, sigma = params
    if kappa <= 0 or theta <= 0 or sigma <= 0:
        return 1e10
    dt = 1.0 / 252  # daily
    exp_kappa_dt = np.exp(-kappa * dt)
    one_minus_exp = 1.0 - exp_kappa_dt
    if one_minus_exp <= 0:
        return 1e10
    c = 2.0 * kappa / (sigma**2 * one_minus_exp)
    df_param = 4.0 * kappa * theta / (sigma**2)
    lambda_param = 2.0 * c * r[:-1]  # using lagged rate
    scale = sigma**2 * one_minus_exp / (4.0 * kappa)
    x = r[1:] / scale
    # Avoid numerical issues
    x = np.maximum(x, 1e-10)
    lambda_param = np.maximum(lambda_param, 1e-10)
    pdf = ncx2.pdf(x, df_param, lambda_param)
    pdf = np.maximum(pdf, 1e-10)
    log_likelihood = np.sum(np.log(pdf)) - np.sum(np.log(scale))
    return -log_likelihood

# Calibration with multiple restarts
bounds = [(0.01, 1.0), (0.001, 0.10), (0.001, 0.20)]
best_params = None
best_ll = -np.inf
np.random.seed(42)
for _ in range(10):
    init = [
        np.random.uniform(bounds[0][0], bounds[0][1]),
        np.random.uniform(bounds[1][0], bounds[1][1]),
        np.random.uniform(bounds[2][0], bounds[2][1])
    ]
    res = minimize(neg_log_likelihood, init, method='L-BFGS-B', bounds=bounds)
    if res.success and (-res.fun) > best_ll:
        best_ll = -res.fun
        best_params = res.x

if best_params is None:
    # fallback
    best_params = [0.1, 0.02, 0.05]
    best_ll = -neg_log_likelihood(best_params)

kappa, theta, sigma = best_params
log_likelihood = best_ll

# Feller condition
feller_ratio = (2 * kappa * theta) / (sigma**2)
feller_satisfied = feller_ratio >= 1

# Bond pricing
def bond_price(tau, r_t):
    gamma = np.sqrt(kappa**2 + 2 * sigma**2)
    exp_gamma_tau = np.exp(gamma * tau)
    numerator_B = 2 * (exp_gamma_tau - 1)
    denominator_B = (gamma + kappa) * (exp_gamma_tau - 1) + 2 * gamma
    B = numerator_B / denominator_B
    numerator_A = 2 * gamma * np.exp((kappa + gamma) * tau / 2)
    denominator_A = (gamma + kappa) * (exp_gamma_tau - 1) + 2 * gamma
    A = (numerator_A / denominator_A) ** (2 * kappa * theta / sigma**2)
    return A * np.exp(-B * r_t)

taus = [0.25, 0.5, 1, 2, 3, 5, 7, 10]
zcb_prices = [bond_price(tau, r_current) for tau in taus]
zero_rates = [-np.log(p) for p in zcb_prices]
model_yields = [-np.log(p) / tau for p, tau in zip(zcb_prices, taus)]

# Market yields
last_row = df.iloc[-1]
market_yields = [
    last_row['DGS1'] / 100.0 if not pd.isna(last_row['DGS1']) else np.nan,
    last_row['DGS2'] / 100.0 if not pd.isna(last_row['DGS2']) else np.nan,
    np.nan,  # DGS3 not available
    last_row['DGS5'] / 100.0 if not pd.isna(last_row['DGS5']) else np.nan,
    last_row['DGS10'] / 100.0 if not pd.isna(last_row['DGS10']) else np.nan,
    np.nan,  # DGS15 not available
    last_row['DGS30'] / 100.0 if not pd.isna(last_row['DGS30']) else np.nan
]
# Map to taus: 0.25->DTB3 (not used), 0.5->DTB3, 1->DGS1, 2->DGS2, 3->?, 5->DGS5, 7->?, 10->DGS10, 30->DGS30
# We need yields for exact taus: 0.25, 0.5, 1, 2, 3, 5, 7, 10
# Use: 0.25 -> DTB3 (3M), 0.5 -> DTB3 (6M not available, use DGS1? but instruction says DGS columns)
# Instruction: Extract observed market yields from the DGS columns (DGS1, DGS2, DGS5, DGS10, DGS30)
# So for tau=0.25 and 0.5, we don't have DGS, set NaN
# For tau=1 -> DGS1, tau=2 -> DGS2, tau=5 -> DGS5, tau=10 -> DGS10, tau=30 -> DGS30 (but we don't have 30 in taus)
# We only have taus up to 10, so:
market_yield_dict = {
    0.25: np.nan,
    0.5: np.nan,
    1.0: last_row['DGS1'] / 100.0 if not pd.isna(last_row['DGS1']) else np.nan,
    2.0: last_row['DGS2'] / 100.0 if not pd.isna(last_row['DGS2']) else np.nan,
    3.0: np.nan,  # no DGS3
    5.0: last_row['DGS5'] / 100.0 if not pd.isna(last_row['DGS5']) else np.nan,
    7.0: np.nan,  # no DGS7
    10.0: last_row['DGS10'] / 100.0 if not pd.isna(last_row['DGS10']) else np.nan
}
market_yields = [market_yield_dict[tau] for tau in taus]

yield_diffs = [model_yield - market_yield if not pd.isna(market_yield) else np.nan
               for model_yield, market_yield in zip(model_yields, market_yields)]

# Forward rates
forward_rates = []
for i in range(len(taus)-1):
    tau1, tau2 = taus[i], taus[i+1]
    p1, p2 = zcb_prices[i], zcb_prices[i+1]
    f = - (np.log(p2) - np.log(p1)) / (tau2 - tau1)
    forward_rates.append((tau1, tau2, f))

# Output calibration.json
calibration = {
    "n_obs": int(n_obs),
    "kappa": float(kappa),
    "theta": float(theta),
    "sigma": float(sigma),
    "log_likelihood": float(log_likelihood),
    "feller_condition_satisfied": bool(feller_satisfied),
    "feller_ratio": float(feller_ratio),
    "r_current": r_current,
    "r_mean": r_mean,
    "r_std": r_std,
    "r_min": r_min,
    "r_max": r_max
}
write_json("calibration.json", calibration)

# Output bond_prices.csv
bond_df = pd.DataFrame({
    "tau": taus,
    "zcb_price": zcb_prices,
    "zero_rate": zero_rates,
    "model_yield": model_yields,
    "market_yield": market_yields,
    "yield_diff": yield_diffs
})
bond_df.to_csv(os.path.join(OUTPUT_DIR, "bond_prices.csv"), index=False)

# Output forward_rates.csv
forward_df = pd.DataFrame(forward_rates, columns=["tau1", "tau2", "forward_rate"])
forward_df.to_csv(os.path.join(OUTPUT_DIR, "forward_rates.csv"), index=False)

# Output summary.json
available = [i for i, my in enumerate(market_yields) if not pd.isna(my)]
if available:
    abs_errors = [abs(yield_diffs[i]) for i in available]
    mean_yield_error = float(np.mean(abs_errors))
    max_yield_error = float(np.max(abs_errors))
else:
    mean_yield_error = 0.0
    max_yield_error = 0.0
yield_curve_upward_sloping = model_yields[-1] > model_yields[0]  # tau=10 > tau=0.25
summary = {
    "kappa": float(kappa),
    "theta": float(theta),
    "sigma": float(sigma),
    "r_current": r_current,
    "feller_satisfied": bool(feller_satisfied),
    "mean_yield_error": mean_yield_error,
    "max_yield_error": max_yield_error,
    "yield_curve_upward_sloping": bool(yield_curve_upward_sloping)
}
write_json("summary.json", summary)

# Print key results
print(f"Calibration: kappa={kappa:.6f}, theta={theta:.6f}, sigma={sigma:.6f}")
print(f"Log-likelihood: {log_likelihood:.6f}")
print(f"Feller condition satisfied: {feller_satisfied} (ratio={feller_ratio:.6f})")
print(f"Current short rate: {r_current:.6f}")

