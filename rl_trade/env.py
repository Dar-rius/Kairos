import pandas as pd
import numpy as np
import gymnasium as gym
import joblib
import random
import torch
from torch import Tensor
from .compute import return_log, calcul_cost, profit_and_loss, reward_func, convert_to_btc, convert_to_usd 
from gymnasium import spaces
from typing import Any
from sklearn.preprocessing import StandardScaler

# ******* ENV **********
class Env():
    def __init__(self, micro_state:pd.DataFrame, macro_state:pd.DataFrame,
                 price:pd.Series, regime_pred:pd.Series=None, change_pred:pd.Series=None,
                 amount_usd:float=100000.0, cost_rate:float=0.001
                 ):
        self.init_usd_amount = amount_usd
        self.cash = self.init_usd_amount
        self.btc_shorted = 0.0
        self.btc_held = 0.0
        self.btc_value = 0.0
        self.micro_scaler = StandardScaler()
        self.micro_state = self.micro_scaler.fit_transform(micro_state)
        self.macro_scaler = joblib.load("./agent/save/macro_scaler.pkl")
        self.macro_state = self.macro_scaler.transform(macro_state)
        self.total_pnl = 0.0
        self.entry_price = 0.0
        self.regime_pred = regime_pred.to_numpy() if regime_pred is not None else None
        self.change_pred = change_pred.to_numpy() if change_pred is not None else None
        self.price = price.to_numpy()
        # Timestep in environment [Day, Start Hour, Last Hour]
        self.time = [0, 0, 23]
        self.cost_rate = cost_rate
        self.size = macro_state.shape[0]
        self.seq = 24
        self.observation_space = [micro_state.shape[1], macro_state.shape[1]]
        self.action_space = 3
        self.p_values_return = [0.0, self.init_usd_amount]
        self.ema_a = 0.0
        self.ema_b = 0.0
        self.dsr_nu = 0.03
        self.step_ = 0
        self.gamma = 2.0
        self.beta = 0.05
        self.alpha = 0.01
        self.prev_pos = 0
        self.pos = 0
        self.day_total = 0

    def _reset_dsr_stats(self):
        self.ema_a = 0.0
        self.ema_b = 0.0
        self.step_ = 0

    def _update_p_values(self):
        self.p_values_return[0] = self.p_values_return[1]
        self.p_values_return[1] = self.calcul_portfolio_value()

    def _rebalance(self, target_pos: int):
        current_pos = 0
        if self.btc_held > 1e-8: current_pos = 1
        elif self.btc_shorted > 1e-8: current_pos = -1

        # Witdraw money from market (HOLD)
        if target_pos == current_pos:
            return

        # Close the position if we was Long
        if current_pos == 1:
            p_btc = self.btc_held * self.btc_value
            fees = p_btc * self.cost_rate
            pnl = (self.btc_value - self.entry_price) * self.btc_held
            self.total_pnl += pnl - fees
            self.cash += p_btc - fees
            self.btc_held = 0.0
        
        # Cover the position if we was short
        elif current_pos == -1:
            needed_to_cover = self.btc_shorted * self.btc_value
            fees = needed_to_cover * self.cost_rate
            pnl = (self.entry_price - self.btc_value) * self.btc_shorted
            self.total_pnl += pnl - fees
            self.cash -= needed_to_cover + fees
            self.btc_shorted = 0.0

        self.entry_price = 0.0
        p_t = self.calcul_portfolio_value()
        
        if target_pos == 1:
            amount_to_use = self.cash * 0.95
            fees = amount_to_use * self.cost_rate
            self.btc_held = (amount_to_use - fees) / self.btc_value
            self.cash -= amount_to_use
            self.entry_price = self.btc_value
        
        #Short market for 90% from Portfolio
        elif target_pos == -1:
            short_value_usd = p_t * 0.9
            fees = short_value_usd * self.cost_rate
            self.cash += short_value_usd - fees
            self.btc_shorted = short_value_usd / self.btc_value
            self.entry_price = self.btc_value
    
    #Liquidate agent if it lose money in short position
    def liquidation(self) -> bool:
        p_t = self.calcul_portfolio_value()
        val_warn = self.init_usd_amount * 0.1
        return p_t < val_warn

    #Reset variables
    def _all_reset(self, train:bool=True):
        if train:
            min_steps_left = 500
            max_macro_idx = self.size - (min_steps_left // 24) - 1
            random_day = random.randint(0, max_macro_idx) if max_macro_idx > 0 else 0
            # Synchronised the Micro and Macro index
            micro_start = random_day * 24
            micro_end = micro_start + 23
            self.time = [random_day, micro_start, micro_end]
            self.day_total = self.size - int(self.time[0])
        else:
            self.time = [0, 0, 23]
            self.day_total = self.size
        self.seq = 0
        # Reset Portfolio Value
        self.p_values_return = [0.0, self.init_usd_amount]
        self.cash = self.init_usd_amount
        self.total_pnl = 0.0
        self.btc_held = 0.0
        self.btc_shorted = 0.0
        self._reset_dsr_stats()

    #shift by one the env timestep from historic dataset
    def _next(self):
        self.time[1] += 1
        self.time[2] += 1
        self.seq += 1
        if self.seq > 23:
            self.time[0] += 1
            self.seq = 0

    def get_pnl(self) -> float:
        pnl = self.total_pnl
        return pnl

    def calcul_portfolio_value(self) -> float:
        value = self.cash + (self.btc_held * self.btc_value) - (self.btc_shorted * self.btc_value)
        return value

    # Create a  set of state group
    def new_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        macro_idx = min(self.time[0], self.size - 1)
        price_idx = min(self.time[2], self.price.shape[0] - 1)
        start = self.time[1]
        end = self.time[2] + 1
        daily_states = self.micro_state[start:end]
        macro_days = self.macro_state[macro_idx]
        self.btc_value = self.price[price_idx]
        current_pos = self.get_current_position_type()
        return daily_states, macro_days, current_pos

    #Get the current action (position)
    def get_current_position_type(self) -> np.ndarray:
        if self.btc_held > 1e-8: return np.array([0.0, 0.0, 1.0]) # Long
        if self.btc_shorted > 1e-8: return np.array([1.0, 0.0, 0.0]) # Short
        return np.array([0.0, 1.0, 0.0])
    
    #Change the trading position from agent's action 
    def change_pos(self, target_pos:int):
        self.pos = target_pos
        self.prev_pos = self.pos

    # Reset all data and choose the new state
    def reset(self, train:bool=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._all_reset(train)
        return self.new_state()

    def convert_to_tensor(self, macro_state:np.ndarray, micro_state:np.ndarray, pos_type:np.ndarray, p_value:float, device:str) -> tuple:
        return (
                torch.tensor(macro_state, dtype=torch.float32, device=device).unsqueeze(0),
                torch.tensor(micro_state, dtype=torch.float32, device=device).unsqueeze(0),
                torch.tensor(pos_type, dtype=torch.long, device=device).unsqueeze(0),
                torch.tensor(p_value, dtype=torch.float32, device=device).unsqueeze(0)
                )

    # The next step
    def step(self, action:Tensor, belief_probs: Tensor) -> Any:
        future_idx = min(self.time[0] + 1, self.size - 1)
        regime_pred = self.regime_pred[future_idx] if self.regime_pred is not None else None
        change_pred = self.change_pred[future_idx] if self.change_pred is not None else None
        target_pos = int(action.item()) - 1
        belief_: np.ndarray = belief_probs.cpu().numpy()
        self.change_pos(target_pos)
        self._rebalance(target_pos)
        self._next()
        next_state = self.new_state()
        self._update_p_values()
        return_ = return_log(self.p_values_return)
        is_liquidated = self.liquidation()
        truncate = False
        reward = 0.0
        if is_liquidated:
            reward = -10.0
            truncate = True
        else:
            #Calcul the reward
            self.step_+=1
            reward, self.ema_a, self.ema_b = reward_func(return_, belief_,
                                                         self.pos, self.prev_pos,
                                                         self.step_, self.dsr_nu,
                                                         self.ema_a, self.ema_b,
                                                         self.alpha, self.gamma, self.beta)
        done = self.time[2] == self.micro_state.shape[0]
        return next_state, reward, regime_pred, change_pred, truncate, done
