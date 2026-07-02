import os
import wandb
import torch
import numpy as np
import pandas as pd
from collections import deque
from rl_trade.env import Env
from rl_trade.compute import calcul_sharpe_ratio, calcul_mdd
from agent.ppo_belief import PPOTrainer
from agent.buffer import Buffer
from agent.model import Agent, MacroHead
from tqdm import tqdm
from visualizer import Visualizer

# Config
DEVICE = "cuda:0" if torch.cuda.is_available() else 'cpu'
DATA_PATH = './data_off/train_test/'
MODEL_PATH = "./agent/save"
PROJECT = 'Kairos'

print(f"Device is: {DEVICE}")

# PPO hyper-param
LR = 3e-5
GAMMA = 0.999
GAE_LAMBDA = 0.95
CLIP_EPS = 0.1
ENT_COEF = 0.001
VALUE_COEF = 0.5
BELIEF_COEF = 0.3
CHANGE_COEF = 0.5

# Load data
micro_states = pd.read_csv(f"{DATA_PATH}price_train.csv").iloc[:, 1:]
macro_states = pd.read_csv(f"{DATA_PATH}metric_train.csv").iloc[:, 1:]
price_series = pd.read_csv(f"{DATA_PATH}price_close_train.csv")["Close"]
regime_series = pd.read_csv(f"{DATA_PATH}regime_train.csv")["regime"]
change_series = pd.read_csv(f"{DATA_PATH}change_train.csv")["change"]

# Training parameters 
TOTAL_TIMESTAMP = 3000000
BATCH_SIZE = 64
ROLLOUT_STEPS = 2048
NUM_UPDATE = TOTAL_TIMESTAMP // ROLLOUT_STEPS

# Initialize classes
env = Env(micro_states, macro_states, price_series, regime_series, change_series, use_scaler=True)
# Visualizer for actions based on his predictions regime
viz = Visualizer()
ACTION_DIM = env.action_space
STATE_DIM = env.observation_space
#Load macro-head wieght
macro_head =  MacroHead(STATE_DIM[1]).to(DEVICE)
macro_head.load_state_dict(torch.load("./agent/save/macro_head.pt", weights_only=True))
agent = Agent(macro_head, STATE_DIM[0], action_dim=ACTION_DIM).to(DEVICE)
trainer = PPOTrainer(agent,
                     lr=LR,
                     gamma=GAMMA,
                     gae_lambda=GAE_LAMBDA,
                     ent_coef=ENT_COEF,
                     value_coef=VALUE_COEF,
                     belief_coef=BELIEF_COEF,
                     change_coef=CHANGE_COEF,
                     device=DEVICE)
buffer = Buffer(ROLLOUT_STEPS, STATE_DIM[0], STATE_DIM[1], DEVICE)

# Wandb configuration
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
wandb.login()

micro_obs, macro_obs, pos_obs = env.reset()
global_step = 0

# Training Loop
with wandb.init(project=PROJECT, config=config) as run:
    for update in tqdm(range(1, NUM_UPDATE + 1)):
        # Variables that stored train historic
        cumulative_reward = 0.0
        cumulative_pnl = 0.0
        past_action = 0
        portfolio_history: deque[float] = deque()
        btc_history: deque[float] = deque()
        action_counts = {0: 0, 1: 0, 2: 0}
        regime_table = wandb.Table(columns=["step", "type", "value"])
        p_value = env.calcul_portfolio_value()
        stop = False

        # Rollout phase
        for step in range(ROLLOUT_STEPS):
            global_step += 1
            macro_t, micro_t, pos_t, p_value_t = env.convert_to_tensor(macro_obs, micro_obs, pos_obs, p_value, DEVICE)
            with torch.inference_mode():
                action_t, log_prob_t, entropy_t, value_t, belief_logits, change_logits, belief_probs, _, _ = agent.get_action_and_value(micro_t, macro_t, pos_t, p_value_t)

            next_obs, reward, target_regime, target_change, truncate, done = env.step(action_t, belief_probs)
            if past_action == action_t:
                action_counts[1] += 1
            else:
                action_counts[int(action_t)] += 1
            past_action = action_t
            done_casted = 1 if done else 0
            belief_casted = belief_probs.squeeze(0).cpu().numpy()
            
            #Insert data in buffer and variables
            buffer.insert(
                micro_state=micro_obs,
                macro_state=macro_obs,
                pos_type=pos_obs,
                action=action_t.item(),
                old_log_prob=log_prob_t,
                reward=reward,
                value=value_t.item(),
                dones=done_casted,
                target_regime=target_regime,
                target_change=target_change,
                beliefs=belief_casted,
                portfolio=p_value
            )
            #Update the historic
            cumulative_reward += reward
            cumulative_pnl += env.get_pnl()
            portfolio_history.append(p_value)
            btc_history.append(env.btc_value)

            if done or truncate:
                micro_obs, macro_obs, pos_obs = env.reset()
                p_value = env.calcul_portfolio_value()
                stop = True
            else:
                micro_obs, macro_obs, pos_obs = next_obs
                p_value = env.calcul_portfolio_value()
        if stop:
            last_value = 0.0
        else:
            #Collect the last critric value
            with torch.inference_mode():
                macro_t, micro_t, pos_t, p_value_t = env.convert_to_tensor(macro_obs, micro_obs, pos_obs, p_value, DEVICE)
                _, _, _, next_value, _, _, _, _, _ = agent.get_action_and_value(micro_t, macro_t, pos_t, p_value_t)
                last_value = next_value.item()

        #Convert list to numpy array
        portfolio_history_np = np.array(portfolio_history)
        btc_history_np = np.array(btc_history) 
        #Calcul the market metrics
        short_pct = (action_counts[0] / ROLLOUT_STEPS) * 100
        hold_pct = (action_counts[1] / ROLLOUT_STEPS) * 100
        buy_pct = (action_counts[2] / ROLLOUT_STEPS) * 100
        sharpe =  calcul_sharpe_ratio(portfolio_history_np)
        mdd = calcul_mdd(portfolio_history_np)

        rewards_list = buffer.rewards
        values_list = buffer.values
        dones_list = buffer.dones
        regime_pred = buffer.beliefs
        regime_truth = buffer.target_regimes
        #Calcul the GAE
        with torch.inference_mode():
            returns, adv, delta = trainer.compute_gae(rewards_list, values_list, last_value, dones_list)
            buffer.insert_returns(returns, adv)
            correct_regimes = (np.argmax(regime_pred, axis=-1) == regime_truth).mean().item()
        
        #Update the weights
        loss, policy_loss, value_loss, belief_loss, change_loss, entropy = trainer.update(buffer, TOTAL_TIMESTAMP, step, BATCH_SIZE)
        #create scatter visualization
        scatter = viz.log_belief_scatter(buffer)
        # Clean buffer
        buffer.clear()
        
        run.log({'Loss': loss,
                 'policy loss': policy_loss,
                 'value loss': value_loss,
                 'belief loss': belief_loss,
                 'change loss': change_loss,
                 'entropy loss': entropy,
                 'reward': cumulative_reward,
                 'PNL': cumulative_pnl,
                 'sharpe ratio': sharpe,
                 'max drawn down': mdd,
                 'hold frenquency': hold_pct,
                 'buy frequency': buy_pct,
                 'short frequency':short_pct,
                 'Belief Space 3D': scatter,
                 'Belief Accuracy': correct_regimes})

#Save model
if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)
torch.save(agent.state_dict(), './agent/save/agent_saved.pt')
torch.save(macro_head.state_dict(), './agent/save/macro_head_postrained.pt')
