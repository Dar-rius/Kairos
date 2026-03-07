from rl_trade.env import Env
from agent.ppo_belief import PPOTrainer, Writer
from agent.buffer import Buffer
from agent.model import Agent, MacroHead
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from rl_trade.compute import calcul_sharpe_ratio, max_dd
from collections import deque

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_PATH = './data_off/train_test/'
print(f"Training on: {DEVICE}")

# Agent Hyperparam
LR = 3e-5
GAMMA = 0.97
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.02
VALUE_COEF = 0.3
BELIEF_COEF = 0.2

# Load Data
hour_df = pd.read_csv(f"{DATA_PATH}price_train.csv").iloc[:, 1:]
macro_df = pd.read_csv(f"{DATA_PATH}metric_train.csv").iloc[:, 1:]
price_series = pd.read_csv(f"{DATA_PATH}price_close_train.csv")["Close"]
state_series = pd.read_csv(f"{DATA_PATH}state_train.csv")["regime"]

TOTAL_TIMESTAMP = 5000000
BATCH_SIZE = 128
ROLLOUT_STEPS = 2048
NUM_UPDATE = TOTAL_TIMESTAMP // ROLLOUT_STEPS
env = Env(hour_df, macro_df, price_series, state_series, use_scaler=True, device=DEVICE)
ACTION_DIM = env.action_space
STATE_DIM = env.observation_space
belief_model =  MacroHead(STATE_DIM[1]).to(DEVICE)
belief_model.load_state_dict(torch.load("./agent/save/belief_head.pt", weights_only=True))
agent = Agent(STATE_DIM[0], action_dim=ACTION_DIM, pretrained_model=belief_model).to(DEVICE)
trainer = PPOTrainer(agent, lr=LR, gamma=GAMMA, gae_lambda=GAE_LAMBDA, ent_coef=ENT_COEF, value_coef=VALUE_COEF, belief_coef=BELIEF_COEF, device=DEVICE)
buffer = Buffer(ROLLOUT_STEPS, STATE_DIM[0], STATE_DIM[1], DEVICE)
writer = Writer("./runs/train/")

# Run env
micro_obs, macro_obs = env.reset()
global_step = 0
# Training Loop
for update in tqdm(range(1, NUM_UPDATE + 1)):
    cumulative_reward = 0.0
    cumulative_pnl = 0.0
    portfolio_value: deque[float] = deque()
    btc_value: deque[float] = deque()
    action_counts = {0: 0, 1: 0, 2: 0}
    # Collecte phase
    for step in range(ROLLOUT_STEPS):
        global_step += 1
        micro_t = micro_obs.unsqueeze(0)
        macro_t = macro_obs.unsqueeze(0)
        action_masked = env.get_action_mask()
        with torch.inference_mode():
            action_t, log_prob_t, entropy_t, value_t, belief_logits, belief_entropy = agent.get_action_and_value(micro_t, macro_t, mask_action=action_masked)

        next_obs, reward, target_regime, truncate, done = env.step(action_t, belief_entropy)
        action_counts[int(action_t)] += 1
        done_casted = torch.tensor(1.0) if done else torch.tensor(0.0)
        buffer.insert(
            micro_state=micro_t,
            macro_state=macro_t,
            action=action_t,
            old_log_prob=log_prob_t,
            reward=reward,
            value=value_t,
            dones = done_casted,
            target_regime=target_regime
        )
        cumulative_reward += reward
        cumulative_pnl += env.get_pnl()
        portfolio_value.append(env.calcul_portfolio_value().item())
        btc_value.append(env.btc_value.item())
        if done or truncate:
            micro_obs, macro_obs = env.reset()
        else:
            micro_obs, macro_obs = next_obs
    # Optimisation phase
    with torch.inference_mode():
        next_micro_t = micro_obs.unsqueeze(0)
        next_macro_t = macro_obs.unsqueeze(0)
        _, _, _, next_value, _, _ = agent.get_action_and_value(next_micro_t, next_macro_t, mask_action=action_masked)
        last_value = torch.tensor([next_value.item()], device=DEVICE)

    hold_pct = (action_counts[0] / ROLLOUT_STEPS) * 100
    buy_pct = (action_counts[1] / ROLLOUT_STEPS) * 100
    sell_pct = (action_counts[2] / ROLLOUT_STEPS) * 100
    sharpe =  calcul_sharpe_ratio(list(portfolio_value))
    mdd = max_dd(portfolio_value)
    rewards_list = buffer.rewards
    values_list = buffer.values
    dones_list = buffer.dones
    returns, adv = trainer.compute_gae(rewards_list, values_list, last_value, dones_list)
    buffer.insert_returns(returns, adv)
    #Compute Belief PPO
    loss, policy_loss, value_loss, belief_loss, entropy = trainer.update(buffer, TOTAL_TIMESTAMP, step, BATCH_SIZE)
    # Clean buffer
    buffer.clear()
    writer.add(global_step, loss, policy_loss, value_loss, belief_loss, entropy, cumulative_reward, cumulative_pnl, sharpe, mdd, hold_pct, buy_pct, sell_pct)

#Save model
torch.save(agent.state_dict(), './agent/save/agent_saved.pt')
torch.save(belief_model.state_dict(), './agent/save/belief_head_1.pt')
writer.close()
