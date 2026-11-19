import json
import os

def main():
    input_path = '/input/environment/data/earnings.json'
    output_dir = '/app/output/.agent_stage'
    output_path = os.path.join(output_dir, 'results.json')
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    companies = data.get('companies', [])
    results = {}
    
    for company in companies:
        ticker = company.get('ticker')
        quarters = company.get('quarters', [])
        quarters_results = []
        surprise_pct_sum = 0.0
        sue_sum = 0.0
        beat_count = 0
        yoy_growth_sum = 0.0
        
        for quarter in quarters:
            period = quarter.get('period')
            actual_eps = quarter.get('actual_eps', 0.0)
            consensus = quarter.get('consensus_estimate', 0.0)
            std_est = quarter.get('std_estimate', 0.0)
            prior_eps = quarter.get('prior_year_eps', 0.0)
            
            surprise_dollars = actual_eps - consensus
            if consensus != 0:
                surprise_pct = (surprise_dollars / abs(consensus)) * 100.0
            else:
                surprise_pct = 0.0
            if std_est != 0:
                sue = surprise_dollars / std_est
            else:
                sue = 0.0
            if prior_eps != 0:
                yoy_growth = ((actual_eps - prior_eps) / abs(prior_eps)) * 100.0
            else:
                yoy_growth = 0.0
            if surprise_pct > 2.0:
                classification = 'beat'
            elif surprise_pct < -2.0:
                classification = 'miss'
            else:
                classification = 'meet'
            
            quarter_result = {
                'period': period,
                'surprise_dollars': surprise_dollars,
                'surprise_pct': surprise_pct,
                'sue': sue,
                'yoy_growth': yoy_growth,
                'classification': classification
            }
            quarters_results.append(quarter_result)
            
            surprise_pct_sum += surprise_pct
            sue_sum += sue
            if classification == 'beat':
                beat_count += 1
            yoy_growth_sum += yoy_growth
        
        avg_surprise_pct = surprise_pct_sum / 4.0
        avg_sue = sue_sum / 4.0
        beat_rate = beat_count / 4.0
        avg_yoy_growth = yoy_growth_sum / 4.0
        
        summary = {
            'avg_surprise_pct': avg_surprise_pct,
            'avg_sue': avg_sue,
            'beat_rate': beat_rate,
            'avg_yoy_growth': avg_yoy_growth
        }
        
        results[ticker] = {
            'quarters': quarters_results,
            'summary': summary
        }
    
    def round_floats(obj):
        if isinstance(obj, dict):
            return {k: round_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [round_floats(i) for i in obj]
        elif isinstance(obj, float):
            return round(obj, 4)
        else:
            return obj
    
    results = round_floats(results)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results written to {output_path}")
    print(f"Processed {len(companies)} companies")
    
    # Self-checks
    with open(output_path, 'r') as f:
        loaded = json.load(f)
    
    errors = []
    for ticker, company_data in loaded.items():
        quarters = company_data.get('quarters', [])
        if len(quarters) != 4:
            errors.append(f"SELFCHECK_FAIL: {ticker} does not have exactly 4 quarters")
        summary = company_data.get('summary', {})
        surprise_pcts = [q.get('surprise_pct') for q in quarters]
        sues = [q.get('sue') for q in quarters]
        yoy_growths = [q.get('yoy_growth') for q in quarters]
        classifications = [q.get('classification') for q in quarters]
        beat_count = sum(1 for c in classifications if c == 'beat')
        expected_avg_surprise_pct = sum(surprise_pcts) / 4.0
        expected_avg_sue = sum(sues) / 4.0
        expected_beat_rate = beat_count / 4.0
        expected_avg_yoy_growth = sum(yoy_growths) / 4.0
        if abs(summary.get('avg_surprise_pct', 0) - expected_avg_surprise_pct) > 1e-5:
            errors.append(f"SELFCHECK_FAIL: {ticker} avg_surprise_pct mismatch")
        if abs(summary.get('avg_sue', 0) - expected_avg_sue) > 1e-5:
            errors.append(f"SELFCHECK_FAIL: {ticker} avg_sue mismatch")
        if abs(summary.get('beat_rate', 0) - expected_beat_rate) > 1e-5:
            errors.append(f"SELFCHECK_FAIL: {ticker} beat_rate mismatch")
        if abs(summary.get('avg_yoy_growth', 0) - expected_avg_yoy_growth) > 1e-5:
            errors.append(f"SELFCHECK_FAIL: {ticker} avg_yoy_growth mismatch")
        for c in classifications:
            if c not in ['beat', 'miss', 'meet']:
                errors.append(f"SELFCHECK_FAIL: {ticker} invalid classification {c}")
        if not (0 <= summary.get('beat_rate', 0) <= 1):
            errors.append(f"SELFCHECK_FAIL: {ticker} beat_rate out of range")
    
    for err in errors:
        print(err)
    if not errors:
        print("All self-checks passed")

if __name__ == '__main__':
    main()

