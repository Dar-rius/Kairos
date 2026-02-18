import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rl_trade.env import Env
from agent.model import Agent, MacroHead
from tqdm import tqdm
from rl_trade.compute import calcul_sharpe_ratio,max_dd

# Config
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
AGENT_PATH = './agent/save/agent_saved.pt'
BELIEF_PATH = './agent/save/belief_head_1.pt'
DATA_PATH = './data_off/train_test/'
# DataFrame 
hour_df = pd.read_csv(f"{DATA_PATH}price_test.csv").iloc[:, 1:]
macro_df = pd.read_csv(f"{DATA_PATH}metric_test.csv").iloc[:, 1:]
price_series = pd.read_csv(f"{DATA_PATH}price_close_test.csv")["Close"]

# Initialization
env = Env(hour_df, macro_df, price_series)
TEST_STEPS = macro_df.shape[0]
ACTION_DIM = env.action_space
STATE_DIM = env.observation_space
n_days = 0

# Load weights
macro_head = MacroHead(STATE_DIM[1]).to(DEVICE)
print(f"Load model from {AGENT_PATH}...")
print(f"Load model from {BELIEF_PATH}...")
macro_head.load_state_dict(torch.load(BELIEF_PATH, weights_only=True, map_location=DEVICE))
agent = Agent(STATE_DIM[0], ACTION_DIM, pretrained_model=macro_head).to(DEVICE)
agent.load_state_dict(torch.load(AGENT_PATH, weights_only=True, map_location=DEVICE))
agent.eval() # IMPORTANT : Met le modèle en mode évaluation (désactive Dropout, etc.)

print("Run the Backtest...")
micro_obs, macro_obs = env.reset(train=False)

# Tracking
portfolio_history = []
price_history = []
actions_history = []
pnl_history = []
sharpes = []
mdd = []
done = False

for _ in tqdm(range(TEST_STEPS)):
    action_mask = env.get_action_mask()
    micro_obs = micro_obs.unsqueeze(0)
    macro_obs = macro_obs.unsqueeze(0)
    with torch.no_grad():
        action_t, _, _, _, _, _ = agent.get_action_and_value(micro_obs, macro_obs, mask_action=action_mask)
    action = action_t.item()
    next_obs, _, _, _, done = env.step(action, entropy_b=None)
    current_val = env.calcul_portfolio_value()
    current_price = env.btc_value
    portfolio_history.append(current_val.item())
    copy_portfolio = portfolio_history.copy()
    price_history.append(current_price.item())
    actions_history.append(action)
    pnl_history.append(env.get_pnl())
    n_days += 1
    if n_days == 365 or n_days == TEST_STEPS:
        portfolio_s = pd.Series(copy_portfolio)
        returns = portfolio_s.pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(365 * 24) 
        sharpes.append(sharpe.item()) 
        mdd.append(max_dd(portfolio_history).item())
        copy_portfolio.clear()
    if done: break
    micro_obs, macro_obs = next_obs
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
sell_idx = [i for i, x in enumerate(actions_history) if x == 2]
# Display the actions
plt.scatter(buy_idx, [price_history[i] for i in buy_idx], marker='^', color='green', label='Buy', s=50)
plt.scatter(sell_idx, [price_history[i] for i in sell_idx], marker='v', color='red', label='Sell', s=50)
plt.title('Trading Strategy (Price BTC)')
plt.legend()
plt.grid(True)
# Sub-graph 2: Porfolio Value
plt.subplot(2, 1, 2)
plt.plot(portfolio_history, label='Portfolio Value ($)', color='blue')
plt.axhline(y=100000, color='r', linestyle='--', label='Initial Capital') # Assumant 100k départ
plt.title('Porfolio Evolution')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('runs/test/backtest_result.png')
