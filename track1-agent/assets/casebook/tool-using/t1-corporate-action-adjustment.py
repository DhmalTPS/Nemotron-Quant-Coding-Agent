import pandas as pd
import numpy as np
import json
import os

# Read inputs
prices_path = '/input/environment/data/prices.csv'
actions_path = '/input/environment/data/corporate_actions.json'

df = pd.read_csv(prices_path, parse_dates=['date'])
df = df.sort_values('date', kind="stable").reset_index(drop=True)

with open(actions_path, 'r') as f:
    data = json.load(f)
actions = data['corporate_actions']

# Sort actions by date
actions_sorted = sorted(actions, key=lambda x: x['date'])

n = len(df)
price_factor = np.ones(n, dtype=np.float64)
vol_factor = np.ones(n, dtype=np.float64)

price_adjustments = []
volume_adjustments = []
total_dividends = 0.0

for act in actions_sorted:
    act_date = pd.Timestamp(act['date'])
    idx = df['date'].searchsorted(act_date, side='left')
    
    if act['type'] == 'split':
        a, b = map(float, act['ratio'].split(':'))
        split_ratio = a / b
        price_adj = 1.0 / split_ratio
        vol_adj = split_ratio
    else:  # dividend
        div = float(act['amount'])
        total_dividends += div
        if idx < n and df.at[idx, 'date'] == act_date:
            close_ex = df.at[idx, 'close']
        else:
            close_ex = df.at[idx-1, 'close']
        price_adj = (close_ex - div) / close_ex
        vol_adj = 1.0
    
    price_adjustments.append(price_adj)
    volume_adjustments.append(vol_adj)
    
    if idx > 0:
        price_factor[:idx] *= price_adj
        vol_factor[:idx] *= vol_adj

df['close_unadjusted'] = df['close']
df['volume_unadjusted'] = df['volume']
df['close_adjusted'] = (df['close_unadjusted'] * price_factor).round(4)
df['volume_adjusted'] = (df['volume_unadjusted'] * vol_factor).astype(int)

output_dir = '/app/output/.agent_stage'
os.makedirs(output_dir, exist_ok=True)

# Write adjusted_prices.csv
out_csv = os.path.join(output_dir, 'adjusted_prices.csv')
df[['date', 'close_unadjusted', 'close_adjusted', 'volume_unadjusted', 'volume_adjusted']].to_csv(out_csv, index=False)

# Compute results
n_actions = len(actions_sorted)
cum_price_factor = np.prod(price_adjustments) if price_adjustments else 1.0
cum_vol_factor = np.prod(volume_adjustments) if volume_adjustments else 1.0
first_adj_close = df.at[0, 'close_adjusted']
last_unadj_close = df.iloc[-1]['close_unadjusted']

results = {
    'n_actions_applied': n_actions,
    'cumulative_price_adjustment_factor': round(float(cum_price_factor), 8),
    'cumulative_volume_adjustment_factor': round(float(cum_vol_factor), 4),
    'first_date_adjusted_close': round(float(first_adj_close), 4),
    'last_date_unadjusted_close': round(float(last_unadj_close), 4),
    'total_dividends_per_original_share': round(float(total_dividends), 4)
}

# Write results.json
out_json = os.path.join(output_dir, 'results.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)

# Self-checks
def selfcheck():
    ok = True
    # 1. Row count
    if len(df) != 252:
        print(f"SELFCHECK_FAIL: Row count {len(df)} != 252")
        ok = False
    # 2. Date column identical
    if not df['date'].equals(pd.read_csv(prices_path, parse_dates=['date'])['date']):
        print("SELFCHECK_FAIL: Date column mismatch")
        ok = False
    # 3. Most recent adjusted close equals unadjusted close
    if not np.isclose(df.iloc[-1]['close_adjusted'], df.iloc[-1]['close_unadjusted']):
        print(f"SELFCHECK_FAIL: Last adjusted close {df.iloc[-1]['close_adjusted']} != last unadjusted close {df.iloc[-1]['close_unadjusted']}")
        ok = False
    # 4. Volume after last split unchanged
    split_dates = [pd.Timestamp(a['date']) for a in actions_sorted if a['type'] == 'split']
    if split_dates:
        last_split = max(split_dates)
        idx_last = df['date'].searchsorted(last_split, side='left')
        if idx_last < n:
            vol_slice = df.iloc[idx_last:]['volume_adjusted'] - df.iloc[idx_last:]['volume_unadjusted']
            if not np.all(vol_slice == 0):
                print("SELFCHECK_FAIL: Volume adjusted after last split")
                ok = False
    # 5. Cumulative price factor recompute
    if not np.isclose(cum_price_factor, np.prod(price_adjustments)):
        print("SELFCHECK_FAIL: Cumulative price factor mismatch")
        ok = False
    # 6. Total dividends
    if not np.isclose(total_dividends, sum(float(a['amount']) for a in actions_sorted if a['type'] == 'dividend')):
        print("SELFCHECK_FAIL: Total dividends mismatch")
        ok = False
    if ok:
        print("SELFCHECK_PASS: All checks passed")

selfcheck()

# Print progress
print(f"Processed {n_actions} corporate actions")
print(f"Cumulative price adjustment factor: {cum_price_factor:.8f}")
print(f"Cumulative volume adjustment factor: {cum_vol_factor:.4f}")
print(f"First date adjusted close: {first_adj_close:.4f}")
print(f"Last date unadjusted close: {last_unadj_close:.4f}")
print(f"Total dividends per original share: {total_dividends:.4f}")

