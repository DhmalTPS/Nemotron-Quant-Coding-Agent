import json
import math
import os
from scipy.stats import norm

# Read input
with open('/input/environment/data/params.json', 'r') as f:
    params = json.load(f)

forward_rates = params['forward_rates']
vol = params['vol']
strike = params['strike']
notional = params['notional']
period_length = params['period_length']
discount_factors = params['discount_factors']

n = len(forward_rates)
caplets = []
floorlets = []

for i in range(n):
    T_i = i * period_length
    f = forward_rates[i]
    df = discount_factors[i]
    if i == 0:
        caplet = notional * period_length * df * max(f - strike, 0)
        floorlet = notional * period_length * df * max(strike - f, 0)
    else:
        d1 = (math.log(f / strike) + 0.5 * vol**2 * T_i) / (vol * math.sqrt(T_i))
        d2 = d1 - vol * math.sqrt(T_i)
        N = norm.cdf
        caplet = notional * period_length * df * (f * N(d1) - strike * N(d2))
        floorlet = notional * period_length * df * (strike * N(-d2) - f * N(-d1))
    caplets.append(caplet)
    floorlets.append(floorlet)

cap_price = sum(caplets)
floor_price = sum(floorlets)
swap_value = notional * period_length * sum(df_i * (f_i - strike) for df_i, f_i in zip(discount_factors, forward_rates))
parity_error = cap_price - floor_price - swap_value

results = {
    "cap_price": round(cap_price, 2),
    "floor_price": round(floor_price, 2),
    "caplets": [round(x, 4) for x in caplets],
    "floorlets": [round(x, 4) for x in floorlets],
    "swap_value": round(swap_value, 2),
    "put_call_parity_error": round(parity_error, 6)
}

os.makedirs('/app/output/.agent_stage', exist_ok=True)
with open('/app/output/.agent_stage/results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Self-checks
assert len(caplets) == n
assert len(floorlets) == n
assert abs(cap_price - sum(caplets)) < 1e-9
assert abs(floor_price - sum(floorlets)) < 1e-9
assert abs(swap_value - notional * period_length * sum(df_i * (f_i - strike) for df_i, f_i in zip(discount_factors, forward_rates))) < 1e-9
assert abs(parity_error - (cap_price - floor_price - swap_value)) < 1e-9

with open('/app/output/.agent_stage/results.json', 'r') as f:
    loaded = json.load(f)
required_keys = {"cap_price", "floor_price", "caplets", "floorlets", "swap_value", "put_call_parity_error"}
if set(loaded.keys()) != required_keys:
    print("SELFCHECK_FAIL: Invalid keys in output")
if not all(isinstance(loaded[k], (int, float)) for k in ["cap_price", "floor_price", "swap_value", "put_call_parity_error"]):
    print("SELFCHECK_FAIL: Non-numeric value in scalar fields")
if not all(isinstance(x, (int, float)) for x in loaded["caplets"] + loaded["floorlets"]):
    print("SELFCHECK_FAIL: Non-numeric value in caplets or floorlets")
if len(loaded["caplets"]) != n or len(loaded["floorlets"]) != n:
    print("SELFCHECK_FAIL: Length mismatch in caplets or floorlets")

print("Results written to /app/output/results.json")
print(f"Cap price: {results['cap_price']}")
print(f"Floor price: {results['floor_price']}")
print(f"Swap value: {results['swap_value']}")
print(f"Put-call parity error: {results['put_call_parity_error']}")

