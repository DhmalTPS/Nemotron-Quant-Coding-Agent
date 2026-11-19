import pandas as pd
import numpy as np
import json

# Step 1: Read raw data
df_raw = pd.read_csv('/input/environment/data/returns_raw.csv')
n_rows_raw = len(df_raw)

# Step 2: Strip header whitespace
df_raw.columns = df_raw.columns.str.strip()
expected_cols = ["date", "SPY", "TLT", "GLD", "IEF", "EFA"]
assert list(df_raw.columns) == expected_cols, f"Columns mismatch: {list(df_raw.columns)}"

# Step 3: Parse dates
df_raw["date"] = pd.to_datetime(df_raw["date"], format="mixed").dt.strftime("%Y-%m-%d")

# Step 4: Load trading calendar
cal = pd.read_csv('/input/environment/data/trading_calendar.csv')
cal.columns = cal.columns.str.strip()
cal_dates = set(pd.to_datetime(cal.iloc[:, 0]).dt.strftime("%Y-%m-%d"))

# Step 5: Non-trading-day removal
df = df_raw[df_raw["date"].isin(cal_dates)].copy()
n_non_trading_days_removed = n_rows_raw - len(df)

# Step 6: Duplicate-date removal (keep last)
df_before_dup = df.copy()
df = df.drop_duplicates(subset="date", keep="last")
n_duplicates_removed = len(df_before_dup) - len(df)

# Step 7: Outlier removal (|r| > 0.25)
asset_cols = ["SPY", "TLT", "GLD", "IEF", "EFA"]
outlier_mask = (df[asset_cols].abs() > 0.25).fillna(False).any(axis=1)
df_before_out = df.copy()
df = df[~outlier_mask].copy()
n_outliers_removed = len(df_before_out) - len(df)

# Step 8: Listwise NaN deletion
df_before_nan = df.copy()
df = df.dropna(subset=asset_cols)
n_missing_rows_removed = len(df_before_nan) - len(df)

# Step 9: Sort ascending by date
df = df.sort_values("date", kind="stable").reset_index(drop=True)
n_rows_after_cleaning = len(df)
n_observations_clean = n_rows_after_cleaning

# Step 10: Portfolio returns (equal weight, daily rebalanced)
df["portfolio"] = df[asset_cols].mean(axis=1)

# Step 11: Historical VaR
var_95 = -np.percentile(df["portfolio"], 5, method="linear")
var_99 = -np.percentile(df["portfolio"], 1, method="linear")

# Step 12: Worst day
worst_idx = df["portfolio"].idxmin()
worst_day_return = float(df.loc[worst_idx, "portfolio"])
worst_day_date = str(df.loc[worst_idx, "date"])

# Prepare outputs
results = {
    "n_observations_clean": n_rows_after_cleaning,
    "var_95_portfolio": float(var_95),
    "var_99_portfolio": float(var_99),
    "worst_day_return": worst_day_return,
    "worst_day_date": worst_day_date
}

solution = {
    "cleaning_report": {
        "n_rows_raw": n_rows_raw,
        "n_non_trading_days_removed": n_non_trading_days_removed,
        "n_duplicates_removed": n_duplicates_removed,
        "n_outliers_removed": n_outliers_removed,
        "n_missing_rows_removed": n_missing_rows_removed,
        "n_rows_after_cleaning": n_rows_after_cleaning
    }
}

# Write JSON files
write_json("/app/output/.agent_stage/results.json", results)
write_json("/app/output/.agent_stage/solution.json", solution)

# Print progress
print(f"Raw rows: {n_rows_raw}")
print(f"Non-trading days removed: {n_non_trading_days_removed}")
print(f"Duplicates removed: {n_duplicates_removed}")
print(f"Outliers removed: {n_outliers_removed}")
print(f"Missing rows removed: {n_missing_rows_removed}")
print(f"Cleaned rows: {n_rows_after_cleaning}")
print(f"VaR 95%: {var_95:.6f}")
print(f"VaR 99%: {var_99:.6f}")
print(f"Worst day return: {worst_day_return:.6f}")
print(f"Worst day date: {worst_day_date}")

# Self-checks
if n_observations_clean != n_rows_after_cleaning:
    print("SELFCHECK_FAIL: n_observations_clean != n_rows_after_cleaning")
if var_99 < var_95:
    print("SELFCHECK_FAIL: var_99_portfolio < var_95_portfolio")
if worst_day_return > 0:
    print("SELFCHECK_FAIL: worst_day_return > 0 (should be loss or zero)")
if worst_day_date not in df["date"].values:
    print("SELFCHECK_FAIL: worst_day_date not in cleaned data")
removed_sum = n_non_trading_days_removed + n_duplicates_removed + n_outliers_removed + n_missing_rows_removed
if removed_sum + n_rows_after_cleaning != n_rows_raw:
    print("SELFCHECK_FAIL: sum of removed rows + cleaned rows != raw rows")
if not all(pd.Series(df["date"]).str.match(r'^\d{4}-\d{2}-\d{2}$')):
    print("SELFCHECK_FAIL: some dates not in YYYY-MM-DD format")
if not set(df["date"]).issubset(cal_dates):
    print("SELFCHECK_FAIL: some dates not in trading calendar")

