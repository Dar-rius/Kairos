from rl_trade.env import Env
from agent.ppo import PPOTrainer, Writer
from agent.buffer import Buffer
from agent.model import Agent, MacroHead
from tqdm import tqdm # Barre de progression
import torch
import numpy as np
import pandas as pd
from rl_trade.compute import calcul_sharpe_ratio, max_dd, calcul_trade_metrics

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Training on: {DEVICE}")

# Agent Hyperparam
LR = 4.4135003154399014e-05
GAMMA = 0.9637724785369991
GAE_LAMBDA = 0.9705521951905898
CLIP_EPS = 0.2
ENT_COEF = 0.0015161404860857912
VALUE_COEF = 0.16838631009179422
BELIEF_COEF = 0.013925049228912462

# Load Data
hour_df = pd.read_csv("./data_off/train_test/price_train.csv").iloc[:, 1:]
macro_df = pd.read_csv("./data_off/train_test/metric_train.csv").iloc[:, 1:]
price_series = pd.read_csv("./data_off/train_test/price_close_train.csv")["Close"]
state_series = pd.read_csv("./data_off/train_test/state_train.csv")["state"]

TOTAL_TIMESTAMP = 3000000
BATCH_SIZE = 256
ROLLOUT_STEPS = 2048
NUM_UPDATE = TOTAL_TIMESTAMP // ROLLOUT_STEPS
env = Env(hour_df, macro_df, price_series, state_series)
ACTION_DIM = env.action_space
STATE_DIM = env.observation_space
belief_model =  MacroHead(STATE_DIM[1]).to(DEVICE)
belief_model.load_state_dict(torch.load("./agent/save/belief_head.pt", weights_only=True))
agent = Agent(STATE_DIM[0], action_dim=ACTION_DIM, pretrained_model=belief_model).to(DEVICE)
if hasattr(agent, 'actor'):
    agent.belief_head = torch.jit.script(agent.belief_head)
    agent.actor_layer = torch.jit.script(agent.actor_layer)
    agent.critic = torch.jit.script(agent.critic)
trainer = PPOTrainer(agent, lr=LR, gamma=GAMMA, gae_lambda=GAE_LAMBDA, ent_coef=ENT_COEF, value_coef=VALUE_COEF, belief_coef=BELIEF_COEF)
buffer = Buffer(ROLLOUT_STEPS, STATE_DIM[0], STATE_DIM[1], DEVICE)
writer = Writer("./runs/train/")

# Run env
micro_obs, macro_obs = env.reset()
global_step = 0
# Training Loop
for update in tqdm(range(1, NUM_UPDATE + 1)):
    cumulative_reward: float = 0.0
    cumulative_pnl: float = 0.0
    portfolio_value: list[float] = []
    btc_value: list[float] = []
    # Collecte phase
    for step in range(ROLLOUT_STEPS):
        global_step += 1
        micro_t = micro_obs.unsqueeze(0)
        macro_t = macro_obs.unsqueeze(0)
        action_masked = env.get_action_mask()
        with torch.inference_mode():
            action_t, log_prob_t, entropy_t, value_t, belief_logits, belief_entropy = agent.get_action_and_value(micro_t, macro_t, mask_action=action_masked)

        action = action_t
        value = value_t
        log_prob = log_prob_t
        next_obs, reward, target_regime, truncate, done = env.step(action, belief_entropy)
        buffer.insert(
            micro_state=micro_t,
            macro_state=macro_t,
            action=action_t,
            old_log_prob=log_prob_t,
            reward=reward,
            value=value,
            dones = 1.0 if done else 0.0,
            target_regime=target_regime
        )
        cumulative_reward += reward
        cumulative_pnl += env.get_pnl()
        portfolio_value.append(env.calcul_portfolio_value())
        btc_value.append(env.btc_value)
        if done or truncate:
            micro_obs, macro_obs = env.reset()
        else:
            micro_obs, macro_obs = next_obs
    # Optimisation phase
    with torch.inference_mode():
        next_micro_t = micro_obs.unsqueeze(0)
        next_macro_t = macro_obs.unsqueeze(0)
        _, _, _, next_value, _, _ = agent.get_action_and_value(next_micro_t, next_macro_t, action_masked)
        last_value = torch.tensor([next_value.item()], device=DEVICE)

    sharpe =  calcul_sharpe_ratio(portfolio_value, btc_value)
    expectancy =  calcul_trade_metrics(portfolio_value)
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
    writer.add(global_step, loss, policy_loss, value_loss, belief_loss, entropy, cumulative_reward, cumulative_pnl, sharpe, mdd, expectancy)

#Save model
torch.save(agent.state_dict(), './agent/save/agent_saved.pt')
torch.save(belief_model.state_dict(), './agent/save/belief_head_1.pt')
writer.close()
