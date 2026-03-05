from rl_trade.env import Env
from agent.ppo_belief import PPOTrainer
from agent.buffer import Buffer
from agent.model import Agent, MacroHead
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import optuna
from rl_trade.compute import calcul_sharpe_ratio

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Training on: {DEVICE}")
DATA_PATH = './data_off/train_test/'

def objective(trial):
# Agent Hyperparam
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.99)
    gae_lambda = trial.suggest_float("gae_lambda", 0.95, 0.99)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
    value_coef = trial.suggest_float("value_coef", 0.005, 0.5, log=True)
    belief_coef = trial.suggest_float("belief_coef", 0.005, 0.5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

# Load Data
    hour_df = pd.read_csv(f"{DATA_PATH}price_train.csv").iloc[:, 1:]
    macro_df = pd.read_csv(f"{DATA_PATH}metric_train.csv").iloc[:, 1:]
    price_series = pd.read_csv(f"{DATA_PATH}price_close_train.csv")["Close"]
    state_series = pd.read_csv(f"{DATA_PATH}state_train.csv")["regime"]
    
    TOTAL_TIMESTAMP = 2000000
    ROLLOUT_STEPS = 2048
    env = Env(hour_df, macro_df, price_series, state_series, use_scaler=True, device=DEVICE)
    ACTION_DIM = env.action_space
    STATE_DIM = env.observation_space
    belief_model =  MacroHead(STATE_DIM[1]).to(DEVICE)
    belief_model.load_state_dict(torch.load("./agent/save/belief_head.pt", weights_only=True))
    agent = Agent(STATE_DIM[0], action_dim=ACTION_DIM, pretrained_model=belief_model).to(DEVICE)
    trainer = PPOTrainer(agent, lr=lr, gamma=gamma, gae_lambda=gae_lambda, ent_coef=ent_coef, value_coef=value_coef, belief_coef=belief_coef, device=DEVICE)
    buffer = Buffer(ROLLOUT_STEPS, STATE_DIM[0], STATE_DIM[1], DEVICE)

    # Run env
    sharpes: list[float] = []
    micro_obs, macro_obs = env.reset()
    global_step = 0
    # Training Loop
    for epoch in range(1, 100 + 1):
        cumulative_reward = 0.0
        btc_value: list[float] = []
        portfolio_value: list[float] = []
        # Collecte phase
        for step in range(ROLLOUT_STEPS):
            global_step += 1
            micro_t = micro_obs.unsqueeze(0)
            macro_t = macro_obs.unsqueeze(0)
            action_masked = env.get_action_mask()
            with torch.inference_mode():
                action_t, log_prob_t, _, value_t, _, belief_entropy = agent.get_action_and_value(micro_t, macro_t, mask_action=action_masked)

            next_obs, reward, target_regime, truncate, done = env.step(action_t)
            buffer.insert(
                micro_state=micro_t,
                macro_state=macro_t,
                action=action_t,
                old_log_prob=log_prob_t,
                reward=reward,
                value=value_t,
                dones = 1.0 if done else 0.0,
                target_regime=target_regime
            )
            cumulative_reward += reward
            btc_value.append(env.btc_value.item())
            portfolio_value.append(env.calcul_portfolio_value().item())
            if done or truncate:
                micro_obs, macro_obs = env.reset()
            else:
                micro_obs, macro_obs = next_obs

        # Optimization phase
        with torch.inference_mode():
            next_micro_t = micro_obs.unsqueeze(0)
            next_macro_t = macro_obs.unsqueeze(0)
            _, _, _, next_value, _, _ = agent.get_action_and_value(next_micro_t, next_macro_t, mask_action=action_masked)
            last_value = torch.tensor([next_value.item()], device=DEVICE)

        rewards_list = buffer.rewards
        values_list = buffer.values
        dones_list = buffer.dones
        returns, adv = trainer.compute_gae(rewards_list, values_list, last_value, dones_list)
        buffer.insert_returns(returns, adv)
        #Compute Belief PPO
        trainer.update(buffer, TOTAL_TIMESTAMP, step, batch_size)
        # Clean buffer
        buffer.clear()

        returns_s = pd.Series(portfolio_value).pct_change().dropna()
        if len(returns_s) > 1 and returns_s.std() > 1e-8:
            sharpe_epoch = (returns_s.mean() / returns_s.std()) * np.sqrt(365 * 24)
        else:
            sharpe_epoch = 0.0

        sharpes.append(sharpe_epoch)
        # For optuna
        trial.report(sharpe_epoch, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return np.mean(sharpes[-10:]) if len(sharpes) > 10 else np.mean(sharpes)

study = optuna.create_study(direction = 'maximize',
                            storage="sqlite:///db.sqlite3",
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100)
print(study.best_params)
