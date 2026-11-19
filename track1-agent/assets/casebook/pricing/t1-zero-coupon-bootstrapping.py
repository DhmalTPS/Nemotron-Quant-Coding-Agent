import json
import math
import os

def interpolate_zero_rate(t, known_maturities, known_zero_rates):
    if not known_maturities:
        return 0.0
    if t <= known_maturities[0]:
        return known_zero_rates[0]
    if t >= known_maturities[-1]:
        return known_zero_rates[-1]
    for i in range(len(known_maturities) - 1):
        if known_maturities[i] <= t <= known_maturities[i + 1]:
            t_low = known_maturities[i]
            t_high = known_maturities[i + 1]
            z_low = known_zero_rates[i]
            z_high = known_zero_rates[i + 1]
            if t_high == t_low:
                return z_low
            return z_low + (z_high - z_low) * (t - t_low) / (t_high - t_low)
    return known_zero_rates[-1]

def main():
    with open('/input/environment/data/curve_data.json', 'r') as f:
        data = json.load(f)
    
    maturities = data['maturities']
    par_rates = data['par_rates']
    coupon_freq = data['coupon_freq']
    
    paired = list(zip(maturities, par_rates))
    paired.sort(key=lambda x: x[0])
    maturities_sorted, par_rates_sorted = zip(*paired) if paired else ([], [])
    
    known_maturities = []
    known_zero_rates = []
    discount_factors_full = {}
    zero_rates_full = {}
    forward_rates_full = {}
    
    for T, c in zip(maturities_sorted, par_rates_sorted):
        n = int(T * coupon_freq)
        S = 0.0
        for k in range(1, n):
            t = k / coupon_freq
            if t in known_maturities:
                idx = known_maturities.index(t)
                df_t = math.exp(-known_zero_rates[idx] * t)
            else:
                z_t = interpolate_zero_rate(t, known_maturities, known_zero_rates)
                df_t = math.exp(-z_t * t)
            S += df_t
        df_T = (1.0 - (c / coupon_freq) * S) / (1.0 + c / coupon_freq)
        z_T = -math.log(df_T) / T if T > 0 else 0.0
        
        discount_factors_full[str(T)] = df_T
        zero_rates_full[str(T)] = z_T
        known_maturities.append(T)
        known_zero_rates.append(z_T)
    
    for T in maturities_sorted:
        if T < 1.0:
            fwd = zero_rates_full[str(T)]
        else:
            if T == 1.0:
                fwd = zero_rates_full[str(T)]
            else:
                z_T = zero_rates_full[str(T)]
                target = T - 1.0
                if target in known_maturities:
                    idx = known_maturities.index(target)
                    z_Tprev = known_zero_rates[idx]
                else:
                    z_Tprev = interpolate_zero_rate(target, known_maturities, known_zero_rates)
                fwd = (z_T * T - z_Tprev * target)
        forward_rates_full[str(T)] = fwd
    
    def round_dict(d):
        return {k: round(v, 8) for k, v in d.items()}
    
    discount_factors = round_dict(discount_factors_full)
    zero_rates = round_dict(zero_rates_full)
    forward_rates = round_dict(forward_rates_full)
    
    output = {
        "zero_rates": zero_rates,
        "discount_factors": discount_factors,
        "forward_rates": forward_rates
    }
    
    os.makedirs('/app/output/.agent_stage', exist_ok=True)
    with open('/app/output/.agent_stage/results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Bootstrapping complete.")
    print("Maturities:", maturities_sorted)
    print("Zero rates:", zero_rates)
    print("Discount factors:", discount_factors)
    print("Forward rates:", forward_rates)
    
    # Self-checks
    n = len(maturities_sorted)
    assert len(zero_rates_full) == n, "Mismatch in zero rates length"
    assert len(discount_factors_full) == n, "Mismatch in discount factors length"
    assert len(forward_rates_full) == n, "Mismatch in forward rates length"
    
    for T in maturities_sorted:
        T_str = str(T)
        df = discount_factors_full[T_str]
        zr = zero_rates_full[T_str]
        assert 0 < df <= 1 + 1e-12, f"Discount factor out of range at T={T}: {df}"
        df_check = math.exp(-zr * T)
        assert abs(df - df_check) < 1e-8, f"Discount factor mismatch at T={T}: {df} vs {df_check}"
        
        if T >= 1.0:
            if T == 1.0:
                assert abs(forward_rates_full[T_str] - zr) < 1e-8, f"Forward rate not equal to zero rate at T=1"
            else:
                target = T - 1.0
                if target in known_maturities:
                    idx = known_maturities.index(target)
                    z_Tprev = known_zero_rates[idx]
                else:
                    z_Tprev = interpolate_zero_rate(target, known_maturities, known_zero_rates)
                fwd_check = (zr * T - z_Tprev * target)
                fwd = forward_rates_full[T_str]
                assert abs(fwd - fwd_check) < 1e-8, f"Forward rate formula mismatch at T={T}: {fwd} vs {fwd_check}"
    
    print("All self-checks passed.")

if __name__ == "__main__":
    main()

