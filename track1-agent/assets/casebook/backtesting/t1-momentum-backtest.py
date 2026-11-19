import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os

# Read data
df = pd.read_csv('/input/environment/data/spy_prices.csv', parse_dates=['date'])
df = df.sort_values('date', kind="stable").reset_index(drop=True)

# Compute EMAs
df['EMA50'] = df['adj_close'].ewm(span=50, adjust=False).mean()
df['EMA200'] = df['adj_close'].ewm(span=200, adjust=False).mean()

# Initialize
cash = 100000.0
shares = 0
position = 0
entry_price = 0.0
trades = []
portfolio_values = []
n = len(df)

# Iterate through data
for i in range(n):
    # Signal detection (need previous and current day, both EMAs valid)
    if i >= 1:
        prev_valid = not (pd.isna(df['EMA50'].iloc[i-1]) or pd.isna(df['EMA200'].iloc[i-1]))
        curr_valid = not (pd.isna(df['EMA50'].iloc[i]) or pd.isna(df['EMA200'].iloc[i]))
        if prev_valid and curr_valid:
            # Golden cross
            if position == 0 and df['EMA50'].iloc[i-1] <= df['EMA200'].iloc[i-1] and df['EMA50'].iloc[i] > df['EMA200'].iloc[i]:
                if i < n - 1:  # Ensure next day exists
                    exec_price = df['adj_open'].iloc[i+1]
                    shares_to_buy = int(cash // exec_price)
                    if shares_to_buy > 0:
                        cash -= shares_to_buy * exec_price
                        shares = shares_to_buy
                        entry_price = exec_price
                        position = 1
                        trades.append(['SPY', 'BUY', df['date'].iloc[i].strftime('%Y-%m-%d'),
                                       df['date'].iloc[i+1].strftime('%Y-%m-%d'), exec_price, shares_to_buy, None])
            # Death cross
            elif position == 1 and df['EMA50'].iloc[i-1] >= df['EMA200'].iloc[i-1] and df['EMA50'].iloc[i] < df['EMA200'].iloc[i]:
                if i < n - 1:  # Ensure next day exists
                    exec_price = df['adj_open'].iloc[i+1]
                    cash += shares * exec_price
                    pnl = (exec_price - entry_price) * shares
                    trades.append(['SPY', 'SELL', df['date'].iloc[i].strftime('%Y-%m-%d'),
                                   df['date'].iloc[i+1].strftime('%Y-%m-%d'), exec_price, shares, round(pnl, 2)])
                    shares = 0
                    position = 0
                    entry_price = 0.0
    # Portfolio value
    if position == 1:
        value = cash + shares * df['adj_close'].iloc[i]
    else:
        value = cash
    portfolio_values.append(value)

# Forced close if still holding
if position == 1:
    exec_price = df['adj_close'].iloc[-1]
    cash += shares * exec_price
    pnl = (exec_price - entry_price) * shares
    trades.append(['SPY', 'SELL', df['date'].iloc[-1].strftime('%Y-%m-%d'),
                   df['date'].iloc[-1].strftime('%Y-%m-%d'), exec_price, shares, round(pnl, 2)])
    shares = 0
    position = 0

# Compute daily returns
daily_portfolio = pd.Series(portfolio_values)
daily_returns = daily_portfolio.pct_change()
daily_returns.iloc[0] = 0.0

# Cumulative return
cum_return = (1 + daily_returns).cumprod()

# Performance metrics
num_trading_days = len(daily_returns)
total_return = (cash - 100000) / 100000.0
final_capital = round(cash, 2)
annualized_return = daily_returns.mean() * 252
annualized_volatility = daily_returns.std(ddof=1) * np.sqrt(252)
sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0.0
running_max = cum_return.cummax()
drawdown = (running_max - cum_return) / running_max
max_drawdown = drawdown.max()
calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else 0.0
num_trades = len([t for t in trades if t[1] == 'BUY'])
winning_trades = len([t for t in trades if t[1] == 'SELL' and t[6] is not None and t[6] > 0])
win_rate = winning_trades / num_trades if num_trades > 0 else 0.0
total_pnl = round(sum([t[6] for t in trades if t[1] == 'SELL' and t[6] is not None]), 2)

# Results dictionary
results = {
    "SPY": {
        "total_return": total_return,
        "final_capital": final_capital,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "num_trading_days": num_trading_days
    }
}

# Write results.json
write_json("results.json", results)

# Write trades.csv
trades_df = pd.DataFrame(trades, columns=['ticker', 'type', 'signal_date', 'exec_date', 'price', 'shares', 'pnl'])
trades_df.to_csv('/app/output/.agent_stage/trades.csv', index=False)

# Write daily_portfolio.csv
portfolio_df = pd.DataFrame({
    'date': df['date'].dt.strftime('%Y-%m-%d'),
    'cumulative_return': cum_return.values
})
portfolio_df.to_csv('/app/output/.agent_stage/daily_portfolio.csv', index=False)

# Plotly chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=cum_return, mode='lines', name='Cumulative Return'))
fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Break-even", annotation_position="bottom right")
fig.update_layout(
    title="EMA Crossover Momentum: SPY Cumulative Returns",
    xaxis_title="Date",
    yaxis_title="Cumulative Return",
    hovermode="x unified"
)
fig.write_html('/app/output/.agent_stage/cumulative_returns.html')

# Print progress
print("Backtest completed.")
print(f"Total Return: {total_return:.4f}")
print(f"Final Capital: {final_capital:.2f}")
print(f"Annualized Return: {annualized_return:.4f}")
print(f"Annualized Volatility: {annualized_volatility:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Max Drawdown: {max_drawdown:.4f}")
print(f"Calmar Ratio: {calmar_ratio:.4f}")
print(f"Number of Trades: {num_trades}")
print(f"Win Rate: {win_rate:.4f}")
print(f"Total PnL: {total_pnl:.2f}")
print(f"Number of Trading Days: {num_trading_days}")

# Self-checks
# 1. Results JSON keys
expected_keys = ["total_return", "final_capital", "annualized_return", "annualized_volatility",
                 "sharpe_ratio", "max_drawdown", "calmar_ratio", "num_trades", "win_rate",
                 "total_pnl", "num_trading_days"]
if "SPY" not in results:
    print("SELFCHECK_FAIL: Missing 'SPY' key in results")
else:
    for key in expected_keys:
        if key not in results["SPY"]:
            print(f"SELFCHECK_FAIL: Missing key '{key}' in results['SPY']")

# 2. Num trades consistency
buy_trades = len([t for t in trades if t[1] == 'BUY'])
sell_trades = len([t for t in trades if t[1] == 'SELL'])
if buy_trades != sell_trades:
    print(f"SELFCHECK_FAIL: Mismatch in trades: BUY={buy_trades}, SELL={sell_trades}")
if num_trades != buy_trades:
    print(f"SELFCHECK_FAIL: num_trades ({num_trades}) does not equal BUY trades ({buy_trades})")

# 3. Win rate bounds
if not (0 <= win_rate <= 1):
    print(f"SELFCHECK_FAIL: win_rate ({win_rate}) not in [0,1]")

# 4. Max drawdown bounds
if not (0 <= max_drawdown <= 1):
    print(f"SELFCHECK_FAIL: max_drawdown ({max_drawdown}) not in [0,1]")

# 5. Daily portfolio start
if abs(portfolio_df['cumulative_return'].iloc[0] - 1.0) > 1e-9:
    print(f"SELFCHECK_FAIL: daily_portfolio.csv cumulative_return does not start at 1.0, got {portfolio_df['cumulative_return'].iloc[0]}")

# 6. Trades.csv pnl empty for BUY
for idx, row in trades_df.iterrows():
    if row['type'] == 'BUY' and pd.notna(row['pnl']):
        print(f"SELFCHECK_FAIL: BUY row has non-empty pnl at index {idx}")
    if row['type'] == 'SELL' and pd.isna(row['pnl']):
        print(f"SELFCHECK_FAIL: SELL row has empty pnl at index {idx}")

