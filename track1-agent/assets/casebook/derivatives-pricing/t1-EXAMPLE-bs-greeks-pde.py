import pandas as pd
import numpy as np
from scipy.stats import norm

# Read input
df = pd.read_parquet('/input/environment/data/options.parquet')

# Extract parameters
S = df['S'].values
K = df['K'].values
T = df['T'].values
r = df['r'].values
sigma = df['sigma'].values
option_type = df['option_type'].values

# Precompute common terms
sqrt_T = np.sqrt(T)
log_S_K = np.log(S / K)
d1 = (log_S_K + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
d2 = d1 - sigma * sqrt_T
pdf_d1 = norm.pdf(d1)
cdf_d1 = norm.cdf(d1)
cdf_d2 = norm.cdf(d2)
cdf_neg_d1 = norm.cdf(-d1)
cdf_neg_d2 = norm.cdf(-d2)
discount = np.exp(-r * T)

# Price
call_price = S * cdf_d1 - K * discount * cdf_d2
put_price = K * discount * cdf_neg_d2 - S * cdf_neg_d1
price = np.where(option_type == 'call', call_price, put_price)

# Delta
call_delta = cdf_d1
put_delta = -cdf_neg_d1
delta = np.where(option_type == 'call', call_delta, put_delta)

# Gamma
gamma = pdf_d1 / (S * sigma * sqrt_T)

# Vega
vega = S * sqrt_T * pdf_d1

#Theta (per year)
call_theta = -(S * sigma * pdf_d1) / (2 * sqrt_T) - r * K * discount * cdf_d2
put_theta = -(S * sigma * pdf_d1) / (2 * sqrt_T) + r * K * discount * cdf_neg_d2
theta_year = np.where(option_type == 'call', call_theta, put_theta)
# Convert to per calendar day
theta = theta_year / 365.0

# Build results DataFrame
results = pd.DataFrame({
    'option_id': df['option_id'],
    'price': price,
    'delta': delta,
    'gamma': gamma,
    'vega': vega,
    'theta': theta
})

# Write output
results.to_parquet('/app/output/.agent_stage/results.parquet', index=False)

# Self-checks
failures = []
# price >= 0
if (results['price'] < -1e-10).any():
    failures.append("SELFCHECK_FAIL: price negative")
# delta bounds
call_mask = results.index[option_type == 'call']
put_mask = results.index[option_type == 'put']
if (results.loc[call_mask, 'delta'] <= 0).any() or (results.loc[call_mask, 'delta'] >= 1).any():
    failures.append("SELFCHECK_FAIL: call delta not in (0,1)")
if (results.loc[put_mask, 'delta'] >= 0).any() or (results.loc[put_mask, 'delta'] <= -1).any():
    failures.append("SELFCHECK_FAIL: put delta not in (-1,0)")
# gamma > 0
if (results['gamma'] <= 0).any():
    failures.append("SELFCHECK_FAIL: gamma not positive")
# vega > 0
if (results['vega'] <= 0).any():
    failures.append("SELFCHECK_FAIL: vega not positive")
# NaN check
if results.isnull().any().any():
    failures.append("SELFCHECK_FAIL: NaN in output")
# Row count
if len(results) != 98:
    failures.append(f"SELFCHECK_FAIL: expected 98 rows, got {len(results)}")
# Order preserved
if not (results['option_id'].values == df['option_id'].values).all():
    failures.append("SELFCHECK_FAIL: option_id order not preserved")

for f in failures:
    print(f)

# Print progress
print(f"Processed {len(results)} options")
print(f"Price range: {results['price'].min():.6f} to {results['price'].max():.6f}")

