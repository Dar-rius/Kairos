import torch
import numpy as np
import pandas as pd
import optuna 
import wandb
from config import TrainConfig
from rl_trade.env import Env
from agent.ppo_belief import PPOTrainer
from agent.buffer import Buffer
from agent.model import Agent, MacroHead
from tqdm import tqdm
from collections import deque
from optuna.integration.wandb import WeightsAndBiasesCallback

#Config
train_config = TrainConfig(timestamp=1_000_000)

#Set device to all tensor
torch.set_default_device(train_config.device)

# Optuna function
def objective(trial):
    # Train hyper-params 
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.80, 0.99)
    gae_lambda = trial.suggest_float("gae_lambda", 0.80, 0.99)
    ent_coef = trial.suggest_float("ent_coef", 0.01, 0.9)
    value_coef = trial.suggest_float("value_coef", 0.05, 0.5)
    belief_coef = trial.suggest_float("belief_coef", 0.01, 0.5)
    change_coef = trial.suggest_float("change_coef", 0.01, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    #Initialize classes
    env = Env(train_config.data_train["micro"],
          train_config.data_train["macro"],
          train_config.data_train["price"],
          train_config.data_train["regime"],
          train_config.data_train["change"])
    ACTION_DIM = env.action_space
    STATE_DIM = env.observation_space
    #Load model's params
    macro_head = MacroHead(STATE_DIM[1])
    macro_head.load_state_dict(torch.load("./agent/save/macro_head.pt", weights_only=True))
    agent = Agent(macro_head, STATE_DIM[0], action_dim=ACTION_DIM)
    trainer = PPOTrainer(agent,
                     lr=lr,
                     gamma=gamma,
                     gae_lambda=gae_lambda,
                     ent_coef=ent_coef,
                     value_coef=value_coef,
                     belief_coef=belief_coef,
                     change_coef=change_coef)
    #Initialize Buffer
    buffer = Buffer(train_config.rollout_steps, STATE_DIM[0], STATE_DIM[1])

    micro_obs, macro_obs = env.reset()
    global_step = 0

    # Start training
    for epoch in range(1, train_config.num_update):
        cumulative_reward = 0.0
        rewards_: deque[float] = deque()
        stop = False
        # Rollout phase
        for step in range(train_config.rollout_steps):
            global_step += 1
            macro_t, micro_t, pos_t, p_value_t = env.convert_to_tensor(macro_obs, micro_obs, pos_obs, p_value)
            with torch.no_grad():
                action_t, log_prob_t, entropy_t, value_t, belief_logits, change_logits, belief_probs, _, _ = agent.get_action_and_value(micro_t, macro_t, pos_t, p_value_t)

            next_obs, reward, target_regime, target_change, truncate, done = env.step(action_t, belief_probs)
            done_casted = 1 if done else 0
            
            #Insert data in buffer and variables
            buffer.insert(
                micro_state=micro_t,
                macro_state=macro_t,
                pos_type=pos_t,
                action=action_t.item(),
                old_log_prob=log_prob_t,
                reward=reward,
                value=value_t.item(),
                dones = done_casted,
                target_regime=target_regime,
                target_change=target_change,
                beliefs = int(torch.argmax(belief_probs).item()),
                portfolio = p_value
            )
            cumulative_reward += reward

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

        rewards_list = buffer.rewards
        values_list = buffer.values
        dones_list = buffer.dones
        #Calcul the GAE
        returns, adv = trainer.compute_gae(rewards_list, values_list,
                                           last_value, dones_list)
        buffer.insert_returns(returns, adv)

        #Update agent's weights
        trainer.update(buffer, train_config.timestamp, step, train_config.batch_size)
        # Clean buffer
        buffer.clear()

        # Report train state to optuna
        trial.report(cumulative_reward, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return  np.mean(rewards_)

#Wandb Callback
wandb_kwargs = {
        "project": "Kairos",
        "name": "search-hyperparam-agent"
        }
wandbc = WeightsAndBiasesCallback(metric="reward", wandb_kwargs=wandb_kwargs)

# Create a new datashboard in optuna dashboard
study = optuna.create_study(direction = 'maximize',
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=50, callbacks=[wandbc])
wandb.finish()
print(study.best_params)
