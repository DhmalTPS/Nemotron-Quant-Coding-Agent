import numpy as np
import pandas as pd
from scipy.stats import norm
import os
import json

# Load data
df = pd.read_csv('/input/environment/data/spy_daily.csv')
close = df['close'].values
log_returns = np.log(close[1:] / close[:-1])
n_prices = len(close)
return_mean = np.mean(log_returns)
return_std = np.std(log_returns, ddof=1)
sigma = return_std * np.sqrt(252)
S0 = close[-1]
r = 0.05

# Calibration output
calibration = {
    "n_prices": int(n_prices),
    "sigma": float(sigma),
    "S0": float(S0),
    "return_mean": float(return_mean),
    "return_std": float(return_std)
}
write_json("calibration.json", calibration)

# Option parameters
T_list = [0.5, 1.0]
n_monitoring_list = [4, 12, 52]
moneyness_list = [0.95, 1.00, 1.05]
N_mc = 100000
rng = np.random.default_rng(42)

rows = []
for T in T_list:
    for n_monitoring in n_monitoring_list:
        t_i = np.arange(1, n_monitoring + 1) * T / n_monitoring
        for m in moneyness_list:
            K = S0 * m
            moneyness = K / S0

            # Exact geometric Asian
            mu_log = (r - 0.5 * sigma**2) * np.mean(t_i)
            var_log = (sigma**2 / n_monitoring**2) * np.sum(np.minimum(t_i[:, None], t_i[None, :]))
            sigma_g = np.sqrt(var_log)
            if sigma_g == 0:
                # Avoid overflow in exp
                if mu_log > 700:
                    geo_exact = np.exp(-r * T) * max(S0 * np.exp(700) - K, 0)
                else:
                    geo_exact = np.exp(-r * T) * max(S0 * np.exp(mu_log) - K, 0)
            else:
                d1 = (np.log(S0 / K) + mu_log + var_log) / sigma_g
                d2 = d1 - sigma_g
                forward = S0 * np.exp(mu_log + 0.5 * var_log)
                geo_exact = np.exp(-r * T) * (forward * norm.cdf(d1) - K * norm.cdf(d2))

            # Levy moment-matching
            mu_i = S0 * np.exp(r * t_i)
            C_ij = S0**2 * np.exp(r * (t_i[:, None] + t_i[None, :])) * (np.exp(sigma**2 * np.minimum(t_i[:, None], t_i[None, :])) - 1)
            m1 = np.mean(mu_i)
            m2 = np.mean(C_ij + mu_i[:, None] * mu_i[None, :])
            v = m2 - m1**2
            if m1 <= 0 or v < 0:
                levy_arith = 0.0
            else:
                sigma2_log = np.log(1 + v / m1**2)
                mu_log_levy = np.log(m1) - 0.5 * sigma2_log
                sigma_levy = np.sqrt(sigma2_log)
                if sigma_levy == 0:
                    # Avoid overflow in exp
                    if (r + 0.5 * sigma_levy**2) * T > 700:
                        levy_arith = np.exp(-r * T) * max(S0 * np.exp(700) - K, 0)
                    else:
                        levy_arith = np.exp(-r * T) * max(S0 * np.exp((r + 0.5 * sigma_levy**2) * T) - K, 0)
                else:
                    d1 = (np.log(S0 / K) + (r + 0.5 * sigma_levy**2) * T) / (sigma_levy * np.sqrt(T))
                    d2 = d1 - sigma_levy * np.sqrt(T)
                    forward_levy = S0 * np.exp((r + 0.5 * sigma_levy**2) * T)
                    levy_arith = np.exp(-r * T) * (forward_levy * norm.cdf(d1) - K * norm.cdf(d2))

            # Curran geometric conditioning
            # Reuse mu_log, var_log, sigma_g from geometric
            # Arithmetic variance
            v_arith = np.mean(C_ij)
            # Cov(log G, A)
            cov_logS_S = S0 * np.exp(r * t_i[:, None] + 0.5 * sigma**2 * t_i[:, None]) * (np.exp(sigma**2 * np.minimum(t_i[:, None], t_i[None, :])) - 1)
            cov_logG_A = np.mean(np.mean(cov_logS_S, axis=1))
            beta = cov_logG_A / var_log if var_log > 0 else 0
            sigma_curran = sigma_g * np.sqrt(max(0, 1 - beta**2))
            mu_curran = mu_log + beta * (np.log(m1) - mu_log)
            if sigma_curran == 0:
                # Avoid overflow in exp by checking if mu_curran is too large
                if mu_curran > 700:  # exp(700) overflows in double precision
                    curran_arith = np.exp(-r * T) * max(S0 * np.exp(700) - K, 0)
                else:
                    curran_arith = np.exp(-r * T) * max(S0 * np.exp(mu_curran) - K, 0)
            else:
                # Check for overflow in the exponential
                exponent = mu_curran + 0.5 * sigma_curran**2
                if exponent > 700:
                    forward_curran = S0 * np.exp(700)
                else:
                    forward_curran = S0 * np.exp(exponent)
                d1 = (np.log(S0 / K) + mu_curran + sigma_curran**2) / sigma_curran
                d2 = d1 - sigma_curran
                curran_arith = np.exp(-r * T) * (forward_curran * norm.cdf(d1) - K * norm.cdf(d2))

            # Monte Carlo
            dt = T / n_monitoring
            z = rng.standard_normal(size=(N_mc, n_monitoring))
            increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
            log_returns_path = np.cumsum(increments, axis=1)
            # Avoid overflow in exp by clipping log_returns_path
            log_returns_path = np.clip(log_returns_path, None, 700)
            S_path = S0 * np.exp(log_returns_path)
            A = S_path.mean(axis=1)
            G = np.exp(np.log(S_path).mean(axis=1))
            payoff_arith = np.maximum(A - K, 0)
            payoff_geo = np.maximum(G - K, 0)
            mc_arith = np.exp(-r * T) * np.mean(payoff_arith)
            mc_geo = np.exp(-r * T) * np.mean(payoff_geo)
            mc_std_err = np.exp(-r * T) * np.std(payoff_arith) / np.sqrt(N_mc)

            rows.append([T, n_monitoring, K, moneyness, geo_exact, levy_arith, curran_arith,
                         mc_arith, mc_geo, mc_std_err])

# Create DataFrame and sort
asian_prices_df = pd.DataFrame(rows, columns=['T', 'n_monitoring', 'K', 'moneyness', 'geo_exact',
                                              'levy_arith', 'curran_arith', 'mc_arith', 'mc_geo', 'mc_std_err'])
asian_prices_df = asian_prices_df.sort_values(['T', 'n_monitoring', 'K'], kind="stable").reset_index(drop=True)
asian_prices_df.to_csv(os.path.join(OUTPUT_DIR, 'asian_prices.csv'), index=False)

# Summary statistics
geo_less_than_arith_pct = 100 * np.mean(asian_prices_df['geo_exact'] < asian_prices_df['mc_arith'])
levy_vs_mc_max_error = np.max(np.abs(asian_prices_df['levy_arith'] - asian_prices_df['mc_arith']))
curran_vs_mc_max_error = np.max(np.abs(asian_prices_df['curran_arith'] - asian_prices_df['mc_arith']))
curran_better_than_levy = curran_vs_mc_max_error < levy_vs_mc_max_error
summary = {
    "n_rows": int(len(asian_prices_df)),
    "geo_less_than_arith_pct": float(geo_less_than_arith_pct),
    "levy_vs_mc_max_error": float(levy_vs_mc_max_error),
    "curran_vs_mc_max_error": float(curran_vs_mc_max_error),
    "curran_better_than_levy": bool(curran_better_than_levy)
}
write_json("summary.json", summary)

# Self-checks
if len(asian_prices_df) != 18:
    print("SELFCHECK_FAIL: asian_prices.csv does not have 18 rows")
if summary["n_rows"] != 18:
    print("SELFCHECK_FAIL: summary.json n_rows is not 18")
if not (0 <= summary["geo_less_than_arith_pct"] <= 100):
    print("SELFCHECK_FAIL: geo_less_than_arith_pct out of bounds")
if summary["levy_vs_mc_max_error"] < 0:
    print("SELFCHECK_FAIL: levy_vs_mc_max_error negative")
if summary["curran_vs_mc_max_error"] < 0:
    print("SELFCHECK_FAIL: curran_vs_mc_max_error negative")
if not isinstance(summary["curran_better_than_levy"], bool):
    print("SELFCHECK_FAIL: curran_better_than_levy not bool")

print("Calibration sigma:", sigma)
print("S0:", S0)
print("Number of rows:", len(asian_prices_df))

