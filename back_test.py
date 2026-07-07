import os
import datetime
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import TrainConfig
from rl_trade.env import Env 
from rl_trade.compute import calcul_sharpe_ratio, calcul_mdd
from agent.model import Agent, MacroHead
from tqdm import tqdm
from collections import deque

# Config
train_config = TrainConfig( device='cpu' )
AGENT_PATH = './agent/save/agent_saved.pt'
MACRO_WEIGHTS_PATH = './agent/save/macro_head_1.pt'
GRAPH_PATH = "./runs/test"


# Load data
micro_states = pd.read_csv(f"{train_config.data_path}/daily_test.csv").iloc[:, 1:]
macro_states = pd.read_csv(f"{train_config.data_path}/metric_test.csv").iloc[:, 1:]
price_series = pd.read_csv(f"{train_config.data_path}/price_close_test.csv")["Close"]
INIT_AMOUNT = 10_000.0

# Initialize the environment
env = Env(micro_states, macro_states, price_series, amount_usd=INIT_AMOUNT)
ACTION_DIM = env.action_space
STATE_DIM = env.observation_space
n_days = 0

# Load weights
macro_head = MacroHead(STATE_DIM[1])
print(f"Load model from {AGENT_PATH}...")
print(f"Load model from {MACRO_WEIGHTS_PATH}...")
macro_head.load_state_dict(torch.load(MACRO_WEIGHTS_PATH, weights_only=True))
macro_head.eval()
agent = Agent(macro_head, STATE_DIM[0], action_dim=ACTION_DIM)
agent.load_state_dict(torch.load(AGENT_PATH, weights_only=True))
agent.eval()

print("Run the Backtest...")
micro_obs, macro_obs, pos_obs = env.reset(train=False)

#The tests historic 
portfolio_history : deque[float] = deque()
price_history : deque[float] = deque()
actions_history : deque[int] = deque()
sharpes : deque[float] = deque()
mdd : deque[float] = deque()
pnl_history : deque[float] = deque()
past_action = 0
total_hours = micro_states.shape[0]

# Start inference on environment
for t in tqdm(range(total_hours)):
    btc_val = env.btc_value
    p_value = env.calcul_portfolio_value()
    macro_t, micro_t, pos_t, p_value_t = env.convert_to_tensor(macro_obs, micro_obs, pos_obs, p_value)
    with torch.no_grad():
        action_t, _, _, _, _, _, belief_probs, _, _ = agent.get_action_and_value(micro_t, macro_t, pos_t, p_value_t)
    
    #Next environment step
    next_obs, _, _, _, _, done = env.step(action_t, belief_probs)

    #Stored the historic data
    portfolio_history.append(p_value)
    copy_portfolio = np.array(portfolio_history.copy())
    price_history.append(btc_val)
    action_t -= 1
    if action_t.item() == past_action:
        actions_history.append(3)
    else:
        actions_history.append(action_t.item())
        past_action = action_t.item()
    pnl_history.append(env.get_pnl())

    #Calcul Sharpe Ratio
    if  t % 8760 == 0:
        sr = calcul_sharpe_ratio(copy_portfolio)
        mdd_ = calcul_mdd(copy_portfolio)
        sharpes.append(sr)
        mdd.append(mdd_)
    if done: break
    micro_obs, macro_obs , pos_obs = next_obs

# Display historic data
long_percent = actions_history.count(1) / total_hours
short_percent = actions_history.count(-1) / total_hours
cash_percent = actions_history.count(0) / total_hours
print(f"""Portfolio Final: {portfolio_history[-1]:.2f}$,
      \nPnL Final (Net): {pnl_history[-1]:.2f}$
      \nSharpe Ratio: {sharpes}
      \nMax Drawd Down: {mdd}
      \nAction Frequency: {long_percent}%, {cash_percent}%, {short_percent}%""")

# Plot historic data
plt.figure(figsize=(15, 10))

# Sub-graph 1: BTC price
plt.subplot(2, 1, 1)
plt.plot(price_history, label='BTC Price', color='gray', alpha=0.5)
plt.title('BTC price evolution')
plt.legend()
plt.grid(True)

# Enumerate all actions
buy_idx = [i for i, x in enumerate(actions_history) if x == 1]
short_idx = [i for i, x in enumerate(actions_history) if x == -1]
cash_idx = [i for i, x in enumerate(actions_history) if x == 0]

# Display actions
plt.scatter(buy_idx, [price_history[i] for i in buy_idx], marker='^', color='green', label='Buy', s=50)
plt.scatter(short_idx, [price_history[i] for i in short_idx], marker='v', color='red', label='Short', s=50)
plt.scatter(cash_idx, [price_history[i] for i in cash_idx], marker='x', color='black', label='Cash', s=50)

# Sub-graph 2: Porfolio Value
plt.subplot(2, 1, 2)
plt.plot(portfolio_history, label='Portfolio Value ($)', color='blue')
plt.axhline(y=INIT_AMOUNT, color='r', linestyle='--', label='Initial Capital')
plt.title('Porfolio value evolution')
plt.legend()
plt.grid(True)
plt.tight_layout()

#Save graph
if not os.path.exists(GRAPH_PATH): os.makedirs(GRAPH_PATH)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f'{GRAPH_PATH}/backtest_result_{timestamp}.png')
print(f"Save img: {GRAPH_PATH}/backtest_result_{timestamp}.png")
