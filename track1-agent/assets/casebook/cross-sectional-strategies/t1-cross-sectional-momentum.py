import pandas as pd
import numpy as np
import json
import os

# Load data
df = pd.read_csv('/input/environment/data/prices.csv', parse_dates=['date'])

# Cleaning steps
df = df.sort_values('date', kind="stable")
df = df.drop_duplicates(subset='date', keep='last')
df = df.ffill()
df = df.dropna()
num_clean_rows = len(df)

# Compute simple monthly returns
returns = df.set_index('date').pct_change()
returns = returns.iloc[1:]  # drop first NaN row

# Prepare for momentum signal
stock_cols = ['AAPL', 'AMZN', 'BRK_B', 'GOOGL', 'JNJ', 'JPM', 'META', 'MSFT', 'TSLA', 'XOM']
signal_returns = []
signal_dates = []

# Loop over return months
for t in range(len(returns)):
    if t < 12:
        continue
    # 11-month summed return: t-12 through t-2 inclusive
    window = returns.iloc[t-12:t]  # excludes t-1, includes t-12 to t-2? Wait: t-12 to t-1 exclusive of t-1? Actually need t-12 to t-2 inclusive
    # Correction: slice t-12 to t-1 gives indices t-12,...,t-2 (since end exclusive)
    signal = returns.iloc[t-12:t-1].sum()  # sum of returns from t-12 to t-2
    # Rank stocks by signal descending, tie-break by original column order
    ranked = signal.sort_values(ascending=False, kind="stable")
    # Break ties by original order: assign rank based on order in stock_cols
    # Create a DataFrame for sorting with secondary key
    df_signal = pd.DataFrame({'signal': signal, 'order': range(len(stock_cols))}, index=stock_cols)
    df_signal = df_signal.sort_values(['signal', 'order'], ascending=[False, True], kind="stable")
    ranked_stocks = df_signal.index.tolist()
    longs = ranked_stocks[:3]
    shorts = ranked_stocks[-3:]
    # Portfolio return for month t
    long_ret = returns.iloc[t][longs].mean()
    short_ret = returns.iloc[t][shorts].mean()
    signal_ret = long_ret - short_ret
    signal_returns.append(signal_ret)
    signal_dates.append(returns.index[t])

# Convert to Series
signal_returns = np.array(signal_returns)
signal_dates = pd.Index(signal_dates)
num_signal_months = len(signal_returns)

# Compute cumulative wealth
cumulative = np.cumprod(1 + signal_returns)
cumulative_return = cumulative  # starting from 1.0 implicitly

# Performance metrics
if num_signal_months > 0:
    total_return = np.prod(1 + signal_returns) - 1
    annualized_return = (1 + total_return) ** (12 / num_signal_months) - 1
    annualized_vol = np.std(signal_returns, ddof=1) * np.sqrt(12)
    sharpe = annualized_return / annualized_vol if annualized_vol != 0 else 0.0
    wealth = cumulative_return
    running_max = np.maximum.accumulate(wealth)
    drawdown = (running_max - wealth) / running_max
    max_drawdown = np.max(drawdown) if len(wealth) > 0 else 0.0
else:
    total_return = 0.0
    annualized_return = 0.0
    annualized_vol = 0.0
    sharpe = 0.0
    max_drawdown = 0.0

# Results dictionary
results = {
    'num_clean_rows': int(num_clean_rows),
    'num_signal_months': int(num_signal_months),
    'total_return': float(total_return),
    'annualized_return': float(annualized_return),
    'annualized_vol': float(annualized_vol),
    'sharpe': float(sharpe),
    'max_drawdown': float(max_drawdown)
}

# Write results.json
write_json('results.json', results)

# Write portfolio.csv
portfolio_df = pd.DataFrame({
    'date': signal_dates.strftime('%Y-%m-%d'),
    'signal_return': signal_returns,
    'cumulative_return': cumulative_return
})
portfolio_df.to_csv(os.path.join(OUTPUT_DIR, 'portfolio.csv'), index=False)

# Self-checks
# 1. num_clean_rows matches cleaned price rows
if num_clean_rows != len(df):
    print(f"SELFCHECK_FAIL: num_clean_rows mismatch: {num_clean_rows} vs {len(df)}")
# 2. num_signal_months matches length of signal_returns
if num_signal_months != len(signal_returns):
    print(f"SELFCHECK_FAIL: num_signal_months mismatch: {num_signal_months} vs {len(signal_returns)}")
# 3. portfolio.csv row count matches num_signal_months
if len(portfolio_df) != num_signal_months:
    print(f"SELFCHECK_FAIL: portfolio.csv row count mismatch: {len(portfolio_df)} vs {num_signal_months}")
# 4. dates sorted ascending
dates = pd.to_datetime(portfolio_df['date'])
if not dates.is_monotonic_increasing:
    print("SELFCHECK_FAIL: portfolio.csv dates not sorted ascending")
# 5. cumulative_return[0] approx 1 + signal_return[0] if exists
if num_signal_months > 0:
    expected = 1 + signal_returns[0]
    actual = portfolio_df.iloc[0]['cumulative_return']
    if not np.isclose(expected, actual):
        print(f"SELFCHECK_FAIL: initial cumulative_return mismatch: {expected} vs {actual}")
# 6. max_drawdown between 0 and 1
if not (0 <= max_drawdown <= 1):
    print(f"SELFCHECK_FAIL: max_drawdown out of bounds: {max_drawdown}")
# 7. annualized_vol non-negative
if annualized_vol < 0:
    print(f"SELFCHECK_FAIL: annualized_vol negative: {annualized_vol}")
# 8. sharpe sign matches annualized_return sign if vol > 0
if annualized_vol > 0:
    if (annualized_return > 0 and sharpe < 0) or (annualized_return < 0 and sharpe > 0):
        print(f"SELFCHECK_FAIL: sharpe sign mismatch: return={annualized_return}, sharpe={sharpe}")
# 9. JSON keys exactly as required
expected_keys = {'num_clean_rows', 'num_signal_months', 'total_return', 'annualized_return', 'annualized_vol', 'sharpe', 'max_drawdown'}
actual_keys = set(results.keys())
if actual_keys != expected_keys:
    print(f"SELFCHECK_FAIL: JSON keys mismatch: {actual_keys} vs {expected_keys}")
# 10. CSV columns exactly as required
expected_cols = ['date', 'signal_return', 'cumulative_return']
actual_cols = list(portfolio_df.columns)
if actual_cols != expected_cols:
    print(f"SELFCHECK_FAIL: CSV columns mismatch: {actual_cols} vs {expected_cols}")

# Print key results
print(f"num_clean_rows: {num_clean_rows}")
print(f"num_signal_months: {num_signal_months}")
print(f"total_return: {total_return:.6f}")
print(f"annualized_return: {annualized_return:.6f}")
print(f"annualized_vol: {annualized_vol:.6f}")
print(f"sharpe: {sharpe:.6f}")
print(f"max_drawdown: {max_drawdown:.6f}")

