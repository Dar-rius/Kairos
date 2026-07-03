import os
import wandb
import torch
import numpy as np
import pandas as pd
from config import PPOConfig, TrainConfig, WandbConfig
from collections import deque
from rl_trade.env import Env
from rl_trade.compute import calcul_sharpe_ratio, calcul_mdd
from agent.ppo_belief import PPOTrainer
from agent.buffer import Buffer
from agent.model import Agent, MacroHead
from tqdm import tqdm
from visualizer import Visualizer

#Initt
ppo_config = PPOConfig()
train_config = TrainConfig()
log_config = {
        'epochs': train_config.num_update,
        'lr': ppo_config.lr,
        'gamma': ppo_config.gamma,
        'gae_lambda': ppo_config.gae_lambda,
        'clip_eps': ppo_config.clip_eps,
        'ent_coef': ppo_config.ent_coef,
        'value_coef': ppo_config.value_coef,
        'belief_coef': ppo_config.belief_coef,
        'change_coef': ppo_config.change_coef 
        }
wandb_config = WandbConfig(logs=log_config)

# set all tensor to device
torch.set_default_device(train_config.device)

# Initialize classes
env = Env(train_config.data_train["micro"],
          train_config.data_train["macro"],
          train_config.data_train["price"],
          train_config.data_train["regime"],
          train_config.data_train["change"])
# Visualizer for actions based on his predictions regime
viz = Visualizer()
ACTION_DIM = env.action_space
STATE_DIM = env.observation_space
#Load macro-head wieght
macro_head =  MacroHead(STATE_DIM[1])
macro_head.load_state_dict(torch.load("./agent/save/macro_head.pt", weights_only=True))
agent = Agent(macro_head, STATE_DIM[0], action_dim=ACTION_DIM)
trainer = PPOTrainer(agent,
                     lr=ppo_config.lr,
                     gamma=ppo_config.gamma,
                     gae_lambda=ppo_config.gae_lambda,
                     ent_coef=ppo_config.ent_coef,
                     value_coef=ppo_config.value_coef,
                     belief_coef=ppo_config.belief_coef,
                     change_coef=ppo_config.change_coef)
buffer = Buffer(train_config.rollout_steps, STATE_DIM[0], STATE_DIM[1])

wandb.login()

micro_obs, macro_obs, pos_obs = env.reset()
global_step = 0

# Training Loop
with wandb.init(project=wandb_config.name, config=wandb_config.logs) as run:
    for update in tqdm(range(1, train_config.num_update + 1)):
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
        for step in range(train_config.rollout_steps):
            global_step += 1
            macro_t, micro_t, pos_t, p_value_t = env.convert_to_tensor(macro_obs, micro_obs, pos_obs, p_value)
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
                macro_t, micro_t, pos_t, p_value_t = env.convert_to_tensor(macro_obs, micro_obs, pos_obs, p_value)
                _, _, _, next_value, _, _, _, _, _ = agent.get_action_and_value(micro_t, macro_t, pos_t, p_value_t)
                last_value = next_value.item()

        #Convert list to numpy array
        portfolio_history_np = np.array(portfolio_history)
        btc_history_np = np.array(btc_history) 
        #Calcul the market metrics
        short_pct = (action_counts[0] / train_config.rollout_steps) * 100
        hold_pct = (action_counts[1] / train_config.rollout_steps) * 100
        buy_pct = (action_counts[2] / train_config.rollout_steps) * 100
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
        (loss, policy_loss, value_loss,
         belief_loss, change_loss, entropy) = trainer.update(buffer, train_config.timestamp, step, train_config.batch_size)
        
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
if not os.path.exists(train_config.model_path): os.makedirs(train_config.model_path)
torch.save(agent.state_dict(), './agent/save/agent_saved.pt')
torch.save(macro_head.state_dict(), './agent/save/macro_head_postrained.pt')
