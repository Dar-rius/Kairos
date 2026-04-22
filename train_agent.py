from collections import deque
import os
from kairos.env import Env
from agent.ppo_belief import PPOTrainer
from agent.buffer import Buffer
from agent.model import Agent, MacroHead
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from kairos.compute import calcul_sharpe_ratio, max_dd
import wandb
from visualizer import Visualizer

DEVICE = "cpu"
DATA_PATH = './data_off/train_test/'
MODEL_PATH = "./agent/save"
PROJECT = 'Kairos'
print(f"Training on: {DEVICE}")
wandb.login()

# Agent Hyperparam
LR = 3e-4
GAMMA = 0.995
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.1
VALUE_COEF = 0.3
BELIEF_COEF = 0.4
CHANGE_COEF = 0.3

# Load Data
hour_df = pd.read_csv(f"{DATA_PATH}price_train.csv").iloc[:, 1:]
macro_df = pd.read_csv(f"{DATA_PATH}metric_train.csv").iloc[:, 1:]
price_series = pd.read_csv(f"{DATA_PATH}price_close_train.csv")["Close"]
state_series = pd.read_csv(f"{DATA_PATH}state_train.csv")["regime"]
change_series = pd.read_csv(f"{DATA_PATH}change_train.csv")["change"]

TOTAL_TIMESTAMP = 1000000
BATCH_SIZE = 128
ROLLOUT_STEPS = 2048
NUM_UPDATE = TOTAL_TIMESTAMP // ROLLOUT_STEPS
env = Env(hour_df, macro_df, price_series, state_series, change_series, use_scaler=True, device=DEVICE)
viz = Visualizer()
ACTION_DIM = env.action_space
STATE_DIM = env.observation_space
belief_model =  MacroHead(STATE_DIM[1]).to(DEVICE)
belief_model.load_state_dict(torch.load("./agent/save/macro_head.pt", weights_only=True))
agent = Agent(belief_model, STATE_DIM[0], action_dim=ACTION_DIM).to(DEVICE)
trainer = PPOTrainer(agent, lr=LR, gamma=GAMMA, gae_lambda=GAE_LAMBDA, ent_coef=ENT_COEF, value_coef=VALUE_COEF, belief_coef=BELIEF_COEF,  change_coef=CHANGE_COEF, device=DEVICE)
buffer = Buffer(ROLLOUT_STEPS, STATE_DIM[0], STATE_DIM[1], DEVICE)

#wanbd variable
project = "Kairos"
config = {
        'epochs': NUM_UPDATE,
        'lr': LR,
        'gamma': GAMMA,
        'gae_lambda': GAE_LAMBDA,
        'clip_eps': CLIP_EPS,
        'ent_coef': ENT_COEF,
        'value_coef': VALUE_COEF,
        'belief_coef': BELIEF_COEF,
        'change_coef': CHANGE_COEF
        }

# Run env
micro_obs, macro_obs, pos_obs = env.reset()
global_step = 0
# Training Loop
with wandb.init(project=project, config=config) as run:
    for update in tqdm(range(1, NUM_UPDATE + 1)):
        cumulative_reward = 0.0
        cumulative_pnl = 0.0
        portfolio_value: deque[float] = deque()
        btc_value: deque[float] = deque()
        pos_counts = {-1: 0, 0: 0, 1: 0}
        stop = False
        # Collecte phase
        for step in range(ROLLOUT_STEPS):
            global_step += 1
            micro_t = micro_obs.unsqueeze(0)
            macro_t = macro_obs.unsqueeze(0)
            pos_t = pos_obs.unsqueeze(0)
            action_masked = env.get_action_mask()
            with torch.inference_mode():
                action_t, log_prob_t, entropy_t, value_t, belief_logits, change_logits, belief_probs, _ = agent.get_action_and_value(micro_t, macro_t, pos_t, mask_action=action_masked)

            next_obs, reward, target_regime, target_change, truncate, done = env.step(action_t)
            portfolio_val = env.calcul_portfolio_value()
            
            pos_counts[int(pos_t)] += 1
            done_casted = torch.tensor(1.0) if done else torch.tensor(0.0)
            
            buffer.insert(
                micro_state=micro_t,
                macro_state=macro_t,
                pos_type=pos_t,
                action=action_t,
                old_log_prob=log_prob_t,
                reward=reward,
                value=value_t,
                dones = done_casted,
                target_regime=target_regime,
                target_change=target_change,
                beliefs = belief_probs.squeeze(0),
                portfolio = portfolio_val
            )
            cumulative_reward += reward
            cumulative_pnl += env.get_pnl()
            portfolio_value.append(env.calcul_portfolio_value().item())
            btc_value.append(env.btc_value.item())
            if done or truncate:
                micro_obs, macro_obs, pos_obs = env.reset()
                stop = True
            else:
                micro_obs, macro_obs, pos_obs = next_obs
        if stop:
            last_value = torch.tensor([0.0], device=DEVICE)
        else:
            # Optimisation phase
            with torch.inference_mode():
                next_micro_t = micro_obs.unsqueeze(0)
                next_macro_t = macro_obs.unsqueeze(0)
                pos_t = pos_obs.unsqueeze(0)
                _, _, _, next_value, _, _, _, _ = agent.get_action_and_value(next_micro_t, next_macro_t, pos_t, mask_action=action_masked)
                last_value = torch.tensor([next_value.item()], device=DEVICE)

        short_pct = (pos_counts[-1] / ROLLOUT_STEPS) * 100
        hold_pct = (pos_counts[0] / ROLLOUT_STEPS) * 100
        buy_pct = (pos_counts[1] / ROLLOUT_STEPS) * 100
        sharpe =  calcul_sharpe_ratio(list(portfolio_value))
        mdd = max_dd(portfolio_value)
        rewards_list = buffer.rewards
        values_list = buffer.values
        dones_list = buffer.dones
        returns, adv, delta = trainer.compute_gae(rewards_list, values_list, last_value, dones_list)
        buffer.insert_returns(returns, adv)
        #Compute Belief PPO
        loss, policy_loss, value_loss, belief_loss, change_loss, entropy = trainer.update(buffer, TOTAL_TIMESTAMP, step, BATCH_SIZE)
        trainer.reset_aes()
        #create scatter
        scatter = viz.log_belief_scatter(buffer)
        # Clean buffer
        buffer.clear()
        run.log({'loss': loss,
                 'policy loss': policy_loss,
                 'value loss': value_loss,
                 'belief loss': belief_loss,
                 'change loss': change_loss,
                 'entropy': entropy,
                 'reward': cumulative_reward,
                 'pnl': cumulative_pnl,
                 'sharpe ratio': sharpe,
                 'max drawn down': mdd,
                 'hold frenquency': hold_pct,
                 'buy frequency': buy_pct,
                 'short frequency':short_pct,
                 'Belief Space 3D': scatter})

#Save model
if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)
torch.save(agent.state_dict(), './agent/save/agent_saved.pt')
torch.save(belief_model.state_dict(), './agent/save/macro_head_1.pt')
