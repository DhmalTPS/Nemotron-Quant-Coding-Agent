import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

def black_scholes_price(S, K, T, r, q, sigma, option_type):
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price

def implied_volatility(price, S, K, T, r, q, option_type):
    def objective(sigma):
        return black_scholes_price(S, K, T, r, q, sigma, option_type) - price
    try:
        vol = brentq(objective, 1e-8, 5.0, xtol=1e-8, maxiter=100)
        return vol
    except ValueError:
        return np.nan

def main():
    # Load inputs
    with open('/input/environment/data/params.json', 'r') as f:
        params = json.load(f)
    df = pd.read_csv('/input/environment/data/option_chain.csv')
    
    # Parameters
    spot = params['spot']
    r = params['risk_free_rate']
    q = params['dividend_yield']
    target_expiry = params['target_expiry']
    T = params['expiries'][target_expiry]
    F = params['forward_prices'][target_expiry]
    min_volume = params['filters']['min_volume']
    max_rel_spread = params['filters']['max_relative_spread']
    exclude_zero_bid = params['filters']['exclude_zero_bid']
    
    # Filter to target expiry
    df_target = df[df['expiry'] == target_expiry].copy()
    n_strikes_input = len(df_target)
    
    # Compute mids
    df_target['call_mid'] = (df_target['call_bid'] + df_target['call_ask']) / 2
    df_target['put_mid'] = (df_target['put_bid'] + df_target['put_ask']) / 2
    
    # Determine OTM side
    df_target['otm_is_put'] = df_target['strike'] < F
    df_target['otm_price'] = np.where(df_target['otm_is_put'], df_target['put_mid'], df_target['call_mid'])
    df_target['otm_bid'] = np.where(df_target['otm_is_put'], df_target['put_bid'], df_target['call_bid'])
    df_target['otm_ask'] = np.where(df_target['otm_is_put'], df_target['put_ask'], df_target['call_ask'])
    df_target['otm_mid'] = df_target['otm_price']
    
    # Apply filters
    cond_vol = df_target['volume'] >= min_volume
    cond_bid = df_target['otm_bid'] > 0 if exclude_zero_bid else pd.Series(True, index=df_target.index)
    cond_spread = (df_target['otm_ask'] - df_target['otm_bid']) / df_target['otm_mid'] <= max_rel_spread
    df_kept = df_target[cond_vol & cond_bid & cond_spread].copy()
    df_kept = df_kept.sort_values('strike', kind="stable").reset_index(drop=True)
    n_strikes_kept = len(df_kept)
    n_otm_puts_kept = (df_kept['strike'] < F).sum()
    n_otm_calls_kept = n_strikes_kept - n_otm_puts_kept
    
    # Implied volatility per kept strike
    iv_list = []
    for _, row in df_kept.iterrows():
        K = row['strike']
        price = row['otm_price']
        option_type = 'put' if row['otm_is_put'] else 'call'
        iv = implied_volatility(price, spot, K, T, r, q, option_type)
        iv_list.append(iv)
    df_kept['iv_strike'] = iv_list
    
    # ATM implied volatility at forward
    strikes = df_kept['strike'].values
    ivs = df_kept['iv_strike'].values
    if F < strikes[0]:
        iv_atm = ivs[0]
        K_lo = K_hi = strikes[0]
    elif F > strikes[-1]:
        iv_atm = ivs[-1]
        K_lo = K_hi = strikes[-1]
    else:
        idx_lo = np.searchsorted(strikes, F, side='right') - 1
        idx_hi = idx_lo + 1
        if strikes[idx_lo] == F:
            iv_atm = ivs[idx_lo]
            K_lo = K_hi = strikes[idx_lo]
        else:
            K_lo = strikes[idx_lo]
            K_hi = strikes[idx_hi]
            w = (F - K_lo) / (K_hi - K_lo)
            iv_atm = (1 - w) * ivs[idx_lo] + w * ivs[idx_hi]
    atm_var = iv_atm ** 2
    
    # Fair variance strike via replication
    n = n_strikes_kept
    dK = np.zeros(n)
    if n >= 2:
        dK[0] = strikes[1] - strikes[0]
        dK[-1] = strikes[-1] - strikes[-2]
        for i in range(1, n-1):
            dK[i] = (strikes[i+1] - strikes[i-1]) / 2.0
    else:
        dK[0] = 0.0
    w = dK / (strikes ** 2)
    sum_weighted = np.sum(w * df_kept['otm_price'].values)
    K_var2 = (2 * np.exp(r * T) / T) * sum_weighted
    fair_variance_strike = K_var2
    fair_vol = np.sqrt(K_var2)
    
    # Risk premium and IV difference
    variance_risk_premium = K_var2 - atm_var
    fair_vol_minus_atm_iv = fair_vol - iv_atm
    
    # Scenario P&L
    N_var = 10000
    scenarios = []
    for name, sigma_RV in [('low', 0.18), ('mid', 0.25), ('high', 0.35)]:
        realized_variance = sigma_RV ** 2
        pnl = N_var * (realized_variance - K_var2)
        scenarios.append({
            'name': name,
            'sigma_RV': sigma_RV,
            'realized_variance': realized_variance,
            'pnl': pnl
        })
    
    # Output
    output = {
        'target_expiry': target_expiry,
        'forward_price': F,
        'n_strikes_input': n_strikes_input,
        'n_strikes_kept': n_strikes_kept,
        'n_otm_puts_kept': int(n_otm_puts_kept),
        'n_otm_calls_kept': int(n_otm_calls_kept),
        'fair_variance_strike': float(fair_variance_strike),
        'fair_vol': float(fair_vol),
        'iv_atm_forward': float(iv_atm),
        'atm_var': float(atm_var),
        'variance_risk_premium': float(variance_risk_premium),
        'fair_vol_minus_atm_iv': float(fair_vol_minus_atm_iv),
        'scenarios': scenarios
    }
    write_json('results.json', output)
    
    # Self-checks
    assert n_strikes_input == 41, f"n_strikes_input expected 41, got {n_strikes_input}"
    assert 0 <= n_strikes_kept <= n_strikes_input, f"n_strikes_kept out of bounds: {n_strikes_kept}"
    assert n_otm_puts_kept + n_otm_calls_kept == n_strikes_kept, "OTM put/call count mismatch"
    assert fair_vol > 0, f"fair_vol not positive: {fair_vol}"
    assert iv_atm > 0, f"iv_atm not positive: {iv_atm}"
    assert abs(variance_risk_premium - (fair_vol**2 - atm_var)) < 1e-6, "variance_risk_premium mismatch"
    for sc in scenarios:
        expected_pnl = N_var * (sc['sigma_RV']**2 - K_var2)
        assert abs(sc['pnl'] - expected_pnl) < 1e-6, f"PNL mismatch for scenario {sc['name']}"
    
    # Print key results
    print(f"Target expiry: {target_expiry}")
    print(f"Forward price: {F}")
    print(f"Strikes input: {n_strikes_input}, kept: {n_strikes_kept}")
    print(f"OTM puts kept: {n_otm_puts_kept}, OTM calls kept: {n_otm_calls_kept}")
    print(f"Fair variance strike: {fair_variance_strike}")
    print(f"Fair vol: {fair_vol}")
    print(f"IV atm forward: {iv_atm}")
    print(f"ATM var: {atm_var}")
    print(f"Variance risk premium: {variance_risk_premium}")
    print(f"Fair vol minus ATM IV: {fair_vol_minus_atm_iv}")
    for sc in scenarios:
        print(f"Scenario {sc['name']}: sigma_RV={sc['sigma_RV']}, realized_var={sc['realized_variance']}, pnl={sc['pnl']}")

if __name__ == '__main__':
    main()

