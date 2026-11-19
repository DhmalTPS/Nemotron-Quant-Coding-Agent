import pandas as pd
import numpy as np

# Load data
baa = pd.read_csv('/input/environment/data/BAA.csv', parse_dates=['observation_date'], na_values='.')
aaa = pd.read_csv('/input/environment/data/AAA.csv', parse_dates=['observation_date'], na_values='.')
dgs10 = pd.read_csv('/input/environment/data/DGS10.csv', parse_dates=['observation_date'], na_values='.')
t10yie = pd.read_csv('/input/environment/data/T10YIE.csv', parse_dates=['observation_date'], na_values='.')
tedrate = pd.read_csv('/input/environment/data/TEDRATE.csv', parse_dates=['observation_date'], na_values='.')

# Set index and resample daily to monthly (last valid observation)
def resample_monthly_last(df):
    df = df.set_index('observation_date')
    monthly = df.resample('ME').last()
    monthly.index = monthly.index.to_period('M').to_timestamp()
    return monthly

dgs10_m = resample_monthly_last(dgs10)
t10yie_m = resample_monthly_last(t10yie)
tedrate_m = resample_monthly_last(tedrate)

# Convert AAA and BAA to monthly (already monthly, but ensure index)
aaa_m = aaa.set_index('observation_date')
aaa_m.index = aaa_m.index.to_period('M').to_timestamp()
baa_m = baa.set_index('observation_date')
baa_m.index = baa_m.index.to_period('M').to_timestamp()

# Merge
df = pd.concat([baa_m, aaa_m, dgs10_m, t10yie_m, tedrate_m], axis=1)
df = df.loc['2003-01-01':'2024-12-01']

# Drop months where BAA, AAA, or DGS10 are missing
df = df.dropna(subset=['BAA', 'AAA', 'DGS10'])

# Forward fill TEDRATE after discontinuation
df['TEDRATE'] = df['TEDRATE'].ffill()

# Drop months where T10YIE is missing
df = df.dropna(subset=['T10YIE'])

# Compute spreads
df['baa_treasury_spread'] = df['BAA'] - df['DGS10']
df['default_proxy'] = df['BAA'] - df['AAA']
df['liquidity_proxy'] = df['TEDRATE']
df['inflation_proxy'] = df['T10YIE']

# Prepare for rolling OLS
window = 48
results = []
dates = df.index[window-1:]  # first window ends at index 47 (Dec 2006)

for i in range(window-1, len(df)):
    window_data = df.iloc[i-window+1:i+1]
    Y = window_data['baa_treasury_spread'].values
    X1 = window_data['default_proxy'].values
    X2 = window_data['liquidity_proxy'].values
    X3 = window_data['inflation_proxy'].values
    X = np.column_stack([np.ones(window), X1, X2, X3])
    
    # OLS
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    alpha, beta1, beta2, beta3 = beta
    Yhat = X @ beta
    residuals = Y - Yhat
    ssr = np.sum(residuals**2)
    sst = np.sum((Y - np.mean(Y))**2)
    r_squared = 1 - ssr/sst if sst != 0 else 0
    
    # Components for current month (last in window)
    default_comp = beta1 * X1[-1]
    liquidity_comp = beta2 * X2[-1]
    inflation_comp = beta3 * X3[-1]
    residual = Y[-1] - alpha - default_comp - liquidity_comp - inflation_comp
    
    # Newey-West HAC with Bartlett kernel, m=3
    m = 3
    XTX = X.T @ X
    XTX_inv = np.linalg.inv(XTX)
    # Gamma0
    gamma0 = np.zeros((4,4))
    for t in range(window):
        xt = X[t:t+1].T  # 4x1
        gamma0 += residuals[t]**2 * (xt @ xt.T)
    # Gamma j
    S = gamma0.copy()
    for j in range(1, m+1):
        gammaj = np.zeros((4,4))
        for t in range(j, window):
            xt = X[t:t+1].T
            x_tj = X[t-j:t-j+1].T
            gammaj += residuals[t] * residuals[t-j] * (xt @ x_tj.T)
        w = 1 - j/(m+1)
        S += w * (gammaj + gammaj.T)
    # HAC covariance
    v_hac = XTX_inv @ S @ XTX_inv
    se = np.sqrt(np.diag(v_hac))
    hac_tstat_default = beta1 / se[1] if se[1] != 0 else 0
    hac_tstat_liquidity = beta2 / se[2] if se[2] != 0 else 0
    hac_tstat_inflation = beta3 / se[3] if se[3] != 0 else 0
    
    # Store
    results.append({
        'date': df.index[i],
        'baa_treasury_spread': Y[-1],
        'default_component': default_comp,
        'liquidity_component': liquidity_comp,
        'inflation_component': inflation_comp,
        'residual': residual,
        'r_squared': r_squared,
        'hac_tstat_default': hac_tstat_default,
        'hac_tstat_liquidity': hac_tstat_liquidity,
        'hac_tstat_inflation': hac_tstat_inflation
    })

# Decomposition results
decomp_df = pd.DataFrame(results)
decomp_df['date'] = decomp_df['date'].dt.strftime('%Y-%m-%d')
cols = ['date', 'baa_treasury_spread', 'default_component', 'liquidity_component', 'inflation_component',
        'residual', 'r_squared', 'hac_tstat_default', 'hac_tstat_liquidity', 'hac_tstat_inflation']
decomp_df = decomp_df[cols]
decomp_df = decomp_df.round(6)
decomp_df.to_csv('/app/output/.agent_stage/decomposition_results.csv', index=False)

# Variance decomposition
var_decomp = []
for i in range(window-1, len(df)):
    window_data = df.iloc[i-window+1:i+1]
    Y = window_data['baa_treasury_spread'].values
    X1 = window_data['default_proxy'].values
    X2 = window_data['liquidity_proxy'].values
    X3 = window_data['inflation_proxy'].values
    X = np.column_stack([np.ones(window), X1, X2, X3])
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    _, beta1, beta2, beta3 = beta
    r_squared = decomp_df.iloc[i-(window-1)]['r_squared']
    var_y = np.var(Y, ddof=1)
    pct_default = (beta1 * np.cov(X1, Y, ddof=1)[0,1] / var_y) * 100 if var_y != 0 else 0
    pct_liquidity = (beta2 * np.cov(X2, Y, ddof=1)[0,1] / var_y) * 100 if var_y != 0 else 0
    pct_inflation = (beta3 * np.cov(X3, Y, ddof=1)[0,1] / var_y) * 100 if var_y != 0 else 0
    pct_residual = (1 - r_squared) * 100
    var_decomp.append({
        'date': df.index[i],
        'pct_default': pct_default,
        'pct_liquidity': pct_liquidity,
        'pct_inflation': pct_inflation,
        'pct_residual': pct_residual
    })

var_df = pd.DataFrame(var_decomp)
var_df['date'] = var_df['date'].dt.strftime('%Y-%m-%d')
var_cols = ['date', 'pct_default', 'pct_liquidity', 'pct_inflation', 'pct_residual']
var_df = var_df[var_cols]
var_df = var_df.round(6)
var_df.to_csv('/app/output/.agent_stage/variance_decomposition.csv', index=False)

# Regime breaks
regime_breaks = decomp_df[decomp_df['r_squared'] < 0.50][['date', 'r_squared']]
regime_breaks = regime_breaks.round(6)
regime_breaks.to_csv('/app/output/.agent_stage/regime_breaks.csv', index=False)

# Self-checks
print(f"Decomposition rows: {len(decomp_df)}")
print(f"Variance decomposition rows: {len(var_df)}")
print(f"Regime breaks: {len(regime_breaks)}")

# Check variance decomposition sums to 100
sums = var_df[['pct_default', 'pct_liquidity', 'pct_inflation', 'pct_residual']].sum(axis=1)
if not np.allclose(sums, 100.0, atol=1e-6):
    for idx, diff in enumerate(sums - 100.0):
        if abs(diff) > 1e-6:
            print(f"SELFCHECK_FAIL: Variance decomposition row {idx} sums to {sums.iloc[idx]}, diff {diff}")

# Check regime breaks r_squared < 0.5
if not (regime_breaks['r_squared'] < 0.5).all():
    print("SELFCHECK_FAIL: Some regime breaks have r_squared >= 0.5")

# Check date range
expected_start = pd.Timestamp('2006-12-01')
expected_end = pd.Timestamp('2024-12-01')
actual_start = pd.Timestamp(decomp_df['date'].iloc[0])
actual_end = pd.Timestamp(decomp_df['date'].iloc[-1])
if actual_start != expected_start:
    print(f"SELFCHECK_FAIL: Expected start date {expected_start}, got {actual_start}")
if actual_end != expected_end:
    print(f"SELFCHECK_FAIL: Expected end date {expected_end}, got {actual_end}")

