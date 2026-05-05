from collections import deque
import os
import datetime
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kairos.env import Env
from agent.model import Agent, MacroHead
from tqdm import tqdm
from kairos.compute import calcul_sharpe_ratio,max_dd

# Config
DEVICE = "cpu"
AGENT_PATH = './agent/save/agent_saved.pt'
BELIEF_PATH = './agent/save/macro_head_1.pt'
DATA_PATH = './data_off/train_test/'
GRAPH_PATH = "./runs/test"
# DataFrame
hour_df = pd.read_csv(f"{DATA_PATH}price_test.csv").iloc[:, 1:]
macro_df = pd.read_csv(f"{DATA_PATH}metric_test.csv").iloc[:, 1:]
price_series = pd.read_csv(f"{DATA_PATH}price_close_test.csv")["Close"]
usd_amount = 10000.0

# Initialization
env = Env(hour_df, macro_df, price_series, amount_usd=usd_amount, use_scaler=True, device=DEVICE)
ACTION_DIM = env.action_space
STATE_DIM = env.observation_space
n_days = 0

# Load weights
macro_head = MacroHead(STATE_DIM[1]).to(DEVICE)
print(f"Load model from {AGENT_PATH}...")
print(f"Load model from {BELIEF_PATH}...")
macro_head.load_state_dict(torch.load(BELIEF_PATH, weights_only=True, map_location=DEVICE))
macro_head.eval()
agent = Agent(macro_head, STATE_DIM[0], action_dim=ACTION_DIM).to(DEVICE)
agent.load_state_dict(torch.load(AGENT_PATH, weights_only=True, map_location=DEVICE))
agent.eval()

print("Run the Backtest...")
micro_obs, macro_obs, pos_obs = env.reset(train=True)

# Tracking
portfolio_history : deque[float] = deque()
price_history : deque[float] = deque()
actions_history : deque[int] = deque()
pnl_history : deque[float] = deque()
sharpes : deque[float] = deque()
mdd : deque[float] = deque()
done = False
past_action = 0

for _ in tqdm(range(env.day_total)):
    action_mask = env.get_action_mask()
    micro_obs = micro_obs.unsqueeze(0)
    macro_obs = macro_obs.unsqueeze(0)
    pos_obs = pos_obs.unsqueeze(0)
    with torch.no_grad():
        action_t, _, _, _, _, _, _, _ = agent.get_action_and_value(micro_obs, macro_obs, pos_obs, mask_action=action_mask)
    next_obs, _, _, _, _, done = env.step(action_t)
    current_val: float = env.calcul_portfolio_value().item()
    current_price = env.btc_value.item()
    portfolio_history.append(current_val)
    copy_portfolio = portfolio_history.copy()
    price_history.append(current_price)
    action_t -= 1
    if action_t.item() == past_action:
        actions_history.append(0)
    else:
        actions_history.append(action_t.item())
    past_action = action_t.item()
    pnl_history.append(env.get_pnl())
    n_days += 1
    if n_days % 365 == 0 or n_days == env.day_total:
        portfolio_s = pd.Series(copy_portfolio)
        returns = portfolio_s.pct_change().dropna()
        if returns.std() == 0.0:
            sharpes.append(0.0)
        else:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(365 * 24)
            sharpes.append(sharpe.item())
        mdd.append(max_dd(portfolio_history))
        copy_portfolio.clear()
    if done: break
    micro_obs, macro_obs , pos_obs = next_obs

results_df = pd.DataFrame({
    'portfolio_value': portfolio_history,
    'btc_value': price_history,
})
# Display last history value
print(f"Portfolio Final: {portfolio_history[-1]:.2f}$, \nPnL Final (Net): {pnl_history[-1]:.2f}$ \nSharpe Ratio: {sharpes} \nMax Drawd Down: {mdd}")
# Plot all historic Bloc
plt.figure(figsize=(15, 10))
# Sub-graph 1: Price BTC and Actions
plt.subplot(2, 1, 1)
plt.plot(price_history, label='BTC Price', color='gray', alpha=0.5)
buy_idx = [i for i, x in enumerate(actions_history) if x == 1]
short_idx = [i for i, x in enumerate(actions_history) if x == -1]
# Display the actions
plt.scatter(buy_idx, [price_history[i] for i in buy_idx], marker='^', color='green', label='Buy', s=50)
plt.scatter(short_idx, [price_history[i] for i in short_idx], marker='v', color='red', label='Short', s=50)
plt.title('Trading Strategy (Price BTC)')
plt.legend()
plt.grid(True)
# Sub-graph 2: Porfolio Value
plt.subplot(2, 1, 2)
plt.plot(portfolio_history, label='Portfolio Value ($)', color='blue')
plt.axhline(y=usd_amount, color='r', linestyle='--', label='Initial Capital') # Assumant 100k départ
plt.title('Porfolio Evolution')
plt.legend()
plt.grid(True)
plt.tight_layout()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if not os.path.exists(GRAPH_PATH): os.makedirs(GRAPH_PATH)
plt.savefig(f'{GRAPH_PATH}/backtest_result_{timestamp}.png')
print(f"Save img: {GRAPH_PATH}/backtest_result_{timestamp}.png")
