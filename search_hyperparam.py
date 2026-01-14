from rl_trade.env import Env
from agent.ppo import PPOTrainer
from agent.buffer import Buffer
from agent.model import Agent, MacroHead
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import optuna

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Training on: {DEVICE}")

def objective(trial):
# Agent Hyperparam
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.99)
    gae_lambda = trial.suggest_float("gae_lambda", 0.95, 0.99)
    clip_eps = 0.2
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
    value_coef = trial.suggest_float("value_coef", 0.005, 0.5, log=True)
    belief_coef = trial.suggest_float("belief_coef", 0.005, 0.5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

# Load Data
    hour_df = pd.read_csv("./data_off/train_test/price_train.csv").iloc[:, 1:]
    macro_df = pd.read_csv("./data_off/train_test/metric_train.csv").iloc[:, 1:]
    price_series = pd.read_csv("./data_off/train_test/price_close_train.csv")["Close"]
    state_series = pd.read_csv("./data_off/train_test/state_train.csv")["state"]

    TOTAL_TIMESTAMP = 2000000
    ROLLOUT_STEPS = 2048
    NUM_UPDATE = TOTAL_TIMESTAMP // ROLLOUT_STEPS
    env = Env(hour_df, macro_df, price_series, state_series)
    ACTION_DIM = env.action_space
    STATE_DIM = env.observation_space
    belief_model =  MacroHead(STATE_DIM[1]).to(DEVICE)
    belief_model.load_state_dict(torch.load("./agent/save/belief_head.pt", weights_only=True))
    agent = Agent(STATE_DIM[0], action_dim=ACTION_DIM, pretrained_model=belief_model).to(DEVICE)
    trainer = PPOTrainer(agent, lr=lr, gamma=gamma, gae_lambda=gae_lambda, ent_coef=ent_coef, value_coef=value_coef, belief_coef=belief_coef)
    buffer = Buffer(ROLLOUT_STEPS, STATE_DIM[0], STATE_DIM[1], DEVICE)

# Run env
    micro_obs, macro_obs = env.reset()
    global_step = 0
# Training Loop
    for epoch in range(1, 10 + 1):
        cumulative_reward: float = 0.0
        cumulative_pnl: float = 0.0
        # Collecte phase
        for step in range(ROLLOUT_STEPS):
            global_step += 1
            micro_t = torch.tensor(micro_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            macro_t = torch.tensor(macro_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action_masked = env.get_action_mask()
            with torch.no_grad():
                action_t, log_prob_t, _, value_t, _, belief_entropy = agent.get_action_and_value(micro_t, macro_t, mask_action=action_masked)

            action = action_t.item()
            value = value_t.item()
            next_obs, reward, target_regime, truncate, done = env.step(action, belief_entropy.item())
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
            if done or truncate:
                micro_obs, macro_obs = env.reset()
            else:
                micro_obs, macro_obs = next_obs
        # Optimisation phase
        with torch.no_grad():
            next_micro_t = torch.tensor(micro_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            next_macro_t = torch.tensor(macro_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            _, _, _, next_value, _, _ = agent.get_action_and_value(next_micro_t, next_macro_t, action_masked)
            last_value = next_value.item()

        rewards_list = buffer.rewards.flatten().tolist()
        values_list = buffer.values.flatten().tolist()
        dones_list = buffer.dones.flatten().tolist()
        returns = trainer.compute_gae(rewards_list, values_list, last_value, dones_list)
        buffer.insert_returns(returns)
        #Compute Belief PPO
        trainer.update(buffer, TOTAL_TIMESTAMP, step, batch_size)
        # Clean buffer
        buffer.clear()

        # For optuna
        trial.report(cumulative_reward, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return cumulative_reward


study = optuna.create_study(direction = 'maximize',
                            storage="sqlite:///db.sqlite3",
                            study_name="rl_optimizer",
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100)
print(study.best_params)
