import pandas as pd
import numpy as np
import scipy.stats as stats
import json
import os

# Load data
df = pd.read_csv('/input/environment/data/sp500_daily.csv')
prices = df['close'].values
log_returns = np.log(prices[1:] / prices[:-1])
losses = -log_returns
losses_sorted = np.sort(losses)[::-1]  # descending

k_list = [25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]

hill_results = []
smith_results = []

for k in k_list:
    u = losses_sorted[k]  # (k+1)-th largest
    excesses = losses_sorted[:k] - u
    # Hill estimator
    xi_hill = np.mean(np.log(losses_sorted[:k] / u))
    se_hill = xi_hill / np.sqrt(k)
    ci_lower_hill = xi_hill - 1.96 * se_hill
    ci_upper_hill = xi_hill + 1.96 * se_hill
    hill_results.append([k, xi_hill, se_hill, ci_lower_hill, ci_upper_hill])
    # Smith (GPD MLE)
    params = stats.genpareto.fit(excesses, floc=0)
    xi_smith, _, sigma_smith = params
    # Asymptotic covariance matrix for GPD MLE with known location=0
    # See e.g. Smith (1987) or Coles (2001) for the observed Fisher information
    # For xi > -0.5, the asymptotic variance of xi is:
    #   var(xi) = sigma^2 * (1 + 2*xi) / (k * (1 + 3*xi + xi**2))
    # We use the estimated xi and sigma and the number of excesses k.
    if 1 + 3*xi_smith + xi_smith**2 > 0:
        var_xi = (sigma_smith**2) * (1 + 2*xi_smith) / (k * (1 + 3*xi_smith + xi_smith**2))
    else:
        # Fallback to the observed information via Hessian if the above is not positive definite
        # Compute negative log-likelihood for GPD
        def nll(params):
            xi, sigma = params
            if sigma <= 0 or (1 + xi * excesses / sigma <= 0).any():
                return np.inf
            z = excesses / sigma
            if abs(xi) < 1e-12:
                return k * np.log(sigma) + np.sum(z)
            else:
                return k * np.log(sigma) + (1 + 1/xi) * np.sum(np.log(1 + xi * z))
        from scipy.optimize import minimize
        res = minimize(nll, [xi_smith, sigma_smith], method='L-BFGS-B', bounds=[(None, None), (1e-12, None)])
        if res.success:
            # Approximate Hessian from the optimizer
            from scipy.optimize import approx_fprime
            # Compute Hessian of nll at the optimum
            eps = 1e-4
            xi_opt, sigma_opt = res.x
            # Compute gradient
            def grad(params):
                xi, sigma = params
                if sigma <= 0 or (1 + xi * excesses / sigma <= 0).any():
                    return np.array([np.inf, np.inf])
                z = excesses / sigma
                if abs(xi) < 1e-12:
                    dxi = np.sum(-z**2 / 2 + z)  # derivative for xi=0 limit
                    dsigma = k/sigma - np.sum(z)/sigma
                else:
                    dxi = -k/xi**2 * np.sum(np.log(1 + xi*z)) + (1+1/xi)*np.sum(z/(1+xi*z))
                    dsigma = k/sigma - (1+1/xi)*np.sum(excesses/(sigma*(1+xi*z)))
                return np.array([dxi, dsigma])
            # Compute Hessian via finite difference of gradient
            hess = np.zeros((2,2))
            for i in range(2):
                for j in range(2):
                    def f(p):
                        return grad(p)[j]
                    hess[i,j] = approx_fprime(res.x, f, eps)[i]
            # Covariance is inverse of Hessian
            cov_mat = np.linalg.inv(hess)
            var_xi = cov_mat[0,0]
        else:
            var_xi = np.nan
    se_xi = np.sqrt(var_xi) if var_xi >= 0 else np.nan
    ci_lower_smith = xi_smith - 1.96 * se_xi
    ci_upper_smith = xi_smith + 1.96 * se_xi
    smith_results.append([k, xi_smith, sigma_smith, se_xi, ci_lower_smith, ci_upper_smith])

# Save CSV files
os.makedirs('/app/output/.agent_stage', exist_ok=True)
hill_df = pd.DataFrame(hill_results, columns=['k', 'xi_hill', 'se_hill', 'ci_lower', 'ci_upper'])
smith_df = pd.DataFrame(smith_results, columns=['k', 'xi_smith', 'sigma_smith', 'se_xi', 'ci_lower', 'ci_upper'])
hill_df.to_csv('/app/output/.agent_stage/hill_estimates.csv', index=False)
smith_df.to_csv('/app/output/.agent_stage/smith_estimates.csv', index=False)

# Comparison metrics
xi_hill_vals = hill_df['xi_hill'].values
xi_smith_vals = smith_df['xi_smith'].values
mean_xi_hill = np.mean(xi_hill_vals)
mean_xi_smith = np.mean(xi_smith_vals)
correlation = np.corrcoef(xi_hill_vals, xi_smith_vals)[0, 1]
comparison = {
    'mean_xi_hill': float(mean_xi_hill),
    'mean_xi_smith': float(mean_xi_smith),
    'correlation': float(correlation)
}
write_json('comparison.json', comparison)

# Summary metrics
best_idx = smith_df['se_xi'].idxmin()
best_k = int(smith_df.loc[best_idx, 'k'])
xi_hill_at_best_k = float(hill_df.loc[hill_df['k'] == best_k, 'xi_hill'].iloc[0])
xi_smith_at_best_k = float(smith_df.loc[best_idx, 'xi_smith'])
tail_is_heavy = mean_xi_smith > 0
summary = {
    'best_k': best_k,
    'xi_hill_at_best_k': xi_hill_at_best_k,
    'xi_smith_at_best_k': xi_smith_at_best_k,
    'tail_is_heavy': tail_is_heavy
}
write_json('summary.json', summary)

# Output key results
print('Hill estimates:')
print(hill_df)
print('Smith estimates:')
print(smith_df)
print('Comparison:', comparison)
print('Summary:', summary)

# Self-checks
def fail(msg):
    print('SELFCHECK_FAIL: ' + msg)

# Check CSV files
if not os.path.exists('/app/output/.agent_stage/hill_estimates.csv'):
    fail('hill_estimates.csv missing')
else:
    df_check = pd.read_csv('/app/output/.agent_stage/hill_estimates.csv')
    if len(df_check) != 10:
        fail('hill_estimates.csv row count not 10')
    if list(df_check.columns) != ['k', 'xi_hill', 'se_hill', 'ci_lower', 'ci_upper']:
        fail('hill_estimates.csv columns mismatch')
    if (df_check['se_hill'] < 0).any():
        fail('hill_estimates.csv negative se')
    if (df_check['ci_lower'] > df_check['ci_upper']).any():
        fail('hill_estimates.csv ci_lower > ci_upper')

if not os.path.exists('/app/output/.agent_stage/smith_estimates.csv'):
    fail('smith_estimates.csv missing')
else:
    df_check = pd.read_csv('/app/output/.agent_stage/smith_estimates.csv')
    if len(df_check) != 10:
        fail('smith_estimates.csv row count not 10')
    if list(df_check.columns) != ['k', 'xi_smith', 'sigma_smith', 'se_xi', 'ci_lower', 'ci_upper']:
        fail('smith_estimates.csv columns mismatch')
    if (df_check['se_xi'] < 0).any():
        fail('smith_estimates.csv negative se')
    if (df_check['ci_lower'] > df_check['ci_upper']).any():
        fail('smith_estimates.csv ci_lower > ci_upper')

# Check JSON files
with open('/app/output/.agent_stage/comparison.json', 'r') as f:
    comp = json.load(f)
if set(comp.keys()) != {'mean_xi_hill', 'mean_xi_smith', 'correlation'}:
    fail('comparison.json keys mismatch')
if not isinstance(comp['mean_xi_hill'], (int, float)):
    fail('comparison.json mean_xi_hill not numeric')
if not isinstance(comp['mean_xi_smith'], (int, float)):
    fail('comparison.json mean_xi_smith not numeric')
if not isinstance(comp['correlation'], (int, float)):
    fail('comparison.json correlation not numeric')
if abs(comp['correlation']) > 1:
    fail('comparison.json correlation out of bounds')

with open('/app/output/.agent_stage/summary.json', 'r') as f:
    summ = json.load(f)
if set(summ.keys()) != {'best_k', 'xi_hill_at_best_k', 'xi_smith_at_best_k', 'tail_is_heavy'}:
    fail('summary.json keys mismatch')
if not isinstance(summ['best_k'], int):
    fail('summary.json best_k not int')
if summ['best_k'] not in k_list:
    fail('summary.json best_k not in k_list')
if not isinstance(summ['xi_hill_at_best_k'], (int, float)):
    fail('summary.json xi_hill_at_best_k not numeric')
if not isinstance(summ['xi_smith_at_best_k'], (int, float)):
    fail('summary.json xi_smith_at_best_k not numeric')
if not isinstance(summ['tail_is_heavy'], bool):
    fail('summary.json tail_is_heavy not bool')
# Verify best_k corresponds to minimal se_xi
smith_df_check = pd.read_csv('/app/output/.agent_stage/smith_estimates.csv')
min_se = smith_df_check['se_xi'].min()
best_rows = smith_df_check[smith_df_check['se_xi'] == min_se]
if summ['best_k'] not in best_rows['k'].values:
    fail('summary.json best_k does not have minimal se_xi')
# Verify tail_is_heavy
if summ['tail_is_heavy'] != (comp['mean_xi_smith'] > 0):
    fail('summary.json tail_is_heavy does not match mean_xi_smith > 0')

