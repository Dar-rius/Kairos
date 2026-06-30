import torch
import numpy as np
import pandas as pd
import optuna 
from rl_trade.env import Env
from agent.ppo_belief import PPOTrainer
from agent.buffer import Buffer
from agent.model import Agent, MacroHead
from tqdm import tqdm
from collections import deque


#Config
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Training on: {DEVICE}")
DATA_PATH = './data_off/train_test/'

# Optuna function
def objective(trial):
    # Train hyper-params 
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.80, 0.99)
    gae_lambda = trial.suggest_float("gae_lambda", 0.80, 0.99)
    ent_coef = trial.suggest_float("ent_coef", 0.01, 0.9)
    value_coef = trial.suggest_float("value_coef", 0.05, 0.5)
    belief_coef = trial.suggest_float("belief_coef", 0.01, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # Load Data
    micro_state = pd.read_csv(f"{DATA_PATH}price_train.csv").iloc[:, 1:]
    macro_state = pd.read_csv(f"{DATA_PATH}metric_train.csv").iloc[:, 1:]
    price_series = pd.read_csv(f"{DATA_PATH}price_close_train.csv")["Close"]
    regime_series = pd.read_csv(f"{DATA_PATH}regime_train.csv")["regime"]
    change_series = pd.read_csv(f"{DATA_PATH}change_train.csv")["change"]
    
    # Training parameters
    TOTAL_TIMESTAMP = 6000000
    ROLLOUT_STEPS = 2048
    NUM_UPDATE = TOTAL_TIMESTAMP // ROLLOUT_STEPS

    #Initialize classes
    env = Env(micro_state, macro_state, price_series, regime_series, change_series)
    ACTION_DIM = env.action_space
    STATE_DIM = env.observation_space
    #Load model's params
    macro_head = MacroHead(STATE_DIM[1]).to(DEVICE)
    macro_head.load_state_dict(torch.load("./agent/save/macro_head.pt", weights_only=True))
    agent = Agent(macro_head, STATE_DIM[0], action_dim=ACTION_DIM).to(DEVICE)
    trainer = PPOTrainer(agent, lr=lr, gamma=gamma, gae_lambda=gae_lambda, ent_coef=ent_coef, value_coef=value_coef, belief_coef=belief_coef, device=DEVICE)
    #Initialize Buffer
    buffer = Buffer(ROLLOUT_STEPS, STATE_DIM[0], STATE_DIM[1], DEVICE)

    micro_obs, macro_obs = env.reset()
    global_step = 0

    # Start training
    for epoch in range(1, NUM_UPDATE):
        cumulative_reward = 0.0
        rewards_: deque[float] = deque()
        stop = False
        # Rollout phase
        for step in range(ROLLOUT_STEPS):
            global_step += 1
            macro_t, micro_t, pos_t, p_value_t = env.convert_to_tensor(macro_obs, micro_obs, pos_obs, p_value, DEVICE)
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
                macro_t, micro_t, pos_t, p_value_t = env.convert_to_tensor(macro_obs, micro_obs, pos_obs, p_value, DEVICE)
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
        trainer.update(buffer, TOTAL_TIMESTAMP, step, batch_size)
        # Clean buffer
        buffer.clear()

        # Report train state to optuna
        trial.report(cumulative_reward, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return  np.mean(rewards_)

# Create a new datashboard in optuna dashboard
study = optuna.create_study(direction = 'maximize',
                            storage="sqlite:///db.sqlite3",
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=50, n_jobs=4)
print(study.best_params)
