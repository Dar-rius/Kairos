from rl_trade.env import Env
from agent.ppo import PPOTrainer, Writer
from agent.buffer import Buffer
from agent.model import Agent
from tqdm import tqdm # Barre de progression
import torch
import numpy as np
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {DEVICE}")

# Agent Hyperparam
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VALUE_COEF = 0.5
BELIEF_COEF = 0.5

# Load Data
daily_df = pd.read_csv("./data_off/train_test/norm_price.csv").iloc[:, 1:]
macro_df = pd.read_csv("./data_off/train_test/metric.csv").iloc[:, 1:]
price_series = pd.read_csv("./data_off/train_test/price_close.csv")["Close"]
state_series = pd.read_csv("./data_off/train_test/state.csv")["state"]
 
env = Env(daily_df, macro_df, price_series, state_series)
ACTION_DIM = env.action_space
STATE_DIM = env.observation_space
ENV_SIZE = 4
NUM_STEP = 128
#BATCH_SIZE = 64
UPDATE_EPOCHS =  macro_df.shape[0] // NUM_STEP
print(macro_df.shape[0])
agent = Agent(STATE_DIM[0], STATE_DIM[1], ACTION_DIM).to(DEVICE)
trainer = PPOTrainer(agent, lr=LR, gamma=GAMMA, gae_lambda=GAE_LAMBDA, ent_coef=ENT_COEF, value_coef=VALUE_COEF, belief_coef=BELIEF_COEF)
buffer = Buffer(NUM_STEP, STATE_DIM[0], STATE_DIM[1], DEVICE)
writer = Writer("./runs/train/")

# Run env
micro_obs, macro_obs = env.reset()
global_step = 0
# Training Loop
for update in range(1, UPDATE_EPOCHS + 1):
    cumulative_reward = 0
    cumulative_pnl = 0
    # Collecte phase
    for step in tqdm(range(NUM_STEP)):
        global_step += 1
        micro_t = torch.tensor(micro_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        macro_t = torch.tensor(macro_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        action_masked = env.get_action_mask()
        with torch.no_grad():
            action_t, log_prob_t, entropy_t, value_t, belief_logits, belief_entropy = agent.get_action_and_value(micro_t, macro_t, mask_action=action_masked)

        action = action_t.item()
        value = value_t.item()
        log_prob = log_prob_t.item()
        next_obs, reward, target_regime, done = env.step(action, belief_entropy)
        buffer.insert(
            micro_state=micro_t,
            macro_state=macro_t,
            action=action_t.item(),
            old_log_prob=log_prob_t.item(),
            reward=reward,
            value=value,
            dones = 1.0 if done else 0.0,
            target_regime=target_regime
        )
        cumulative_reward += reward
        cumulative_pnl += env.get_pnl()
        if done: break
        micro_obs, macro_obs = next_obs # next_obs est un tuple (micro, macro)
    # Optimisation phase
    with torch.no_grad():
        next_micro_t = torch.tensor(micro_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        next_macro_t = torch.tensor(macro_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        _, _, _, next_value, _, _ = agent.get_action_and_value(next_micro_t, next_macro_t)
        last_value = next_value.item()

    rewards_list = buffer.rewards.flatten().tolist()
    values_list = buffer.values.flatten().tolist()
    dones_list = buffer.dones.flatten().tolist()
    returns = trainer.compute_gae(rewards_list, values_list, last_value, dones_list)
    buffer.insert_returns(returns)
    #Compute Belief PPO
    loss, policy_loss, value_loss, belief_loss, entropy = trainer.update(buffer)
    # Clean buffer
    buffer.clear()
    writer.add(global_step, loss, policy_loss, value_loss, belief_loss, entropy, cumulative_reward, cumulative_pnl)

#Save model
torch.save(agent.state_dict(), './agent/save/agent_saved.pt')
writer.close()
