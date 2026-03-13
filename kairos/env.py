import pandas as pd
import numpy as np
from .compute import return_log, calcul_cost, calcul_sharpe_ratio, profit_and_loss, reward_func
from .processing import convert_to_btc, convert_to_usd
import torch
from torch import Tensor
import gymnasium as gym
from gymnasium import spaces
from typing import Any
from sklearn.preprocessing import StandardScaler

# ******* ENV **********
class Env():
    def __init__(self, hour_trade:pd.DataFrame, macro_trade:pd.DataFrame,
                 price:pd.Series, state_pred:pd.Series=None,
                 amount_usd:float=100000.0, cost_rate:float=0.001,
                 device:str='cpu', use_scaler:bool=False
                 ):
        self.device = device
        self.init_usd_amount = amount_usd
        self.use_scaler = use_scaler
        if self.use_scaler:
            # Normalized all dataset
            self.micro_scaler = StandardScaler()
            self.macro_scaler = StandardScaler()
            scaled_hour = self.micro_scaler.fit_transform(hour_trade.values)
            scaled_macro = self.macro_scaler.fit_transform(macro_trade.values)
            self.hour_trade = torch.tensor(scaled_hour, dtype=torch.float32, device=self.device)
            self.macro_trade = torch.tensor(scaled_macro, dtype=torch.float32, device=self.device)
        else:
            self.hour_trade = torch.tensor(hour_trade.values, dtype=torch.float32, device=self.device)
            self.macro_trade = torch.tensor(macro_trade.values, dtype=torch.float32, device=self.device)
        # Total PnL [Buy Price, PnL Brut, Fees, PnL Final]
        self.total_pnl = torch.zeros((4,1), dtype=torch.float32, device=self.device)
        self.btc_value = torch.tensor([0], dtype=torch.float32, device=self.device)
        self.hour_trade = torch.tensor(hour_trade.values, dtype=torch.float32, device=self.device)
        self.macro_trade = torch.tensor(macro_trade.values, dtype=torch.float32, device=self.device)
        self.state_pred = torch.tensor(state_pred.values, dtype=torch.int8, device=self.device) if state_pred is not None else None
        self.price = torch.tensor(price.values, dtype=torch.float32, device=self.device)
        # Time for trades [Day, Start Hour, Last Hour]
        self.time = torch.tensor([0, 0, 23], dtype=torch.int32, device=self.device)
        self.total_amount = torch.tensor([self.init_usd_amount, 0.0], dtype=torch.float32, device=self.device)
        self.cost_rate = torch.tensor(cost_rate, device=self.device)
        self.size = self.macro_trade.shape[0]
        self.seq = torch.tensor(24, device=self.device)
        self.observation_space = self.hour_trade.shape[1], self.macro_trade.shape[1]
        self.action_space = 3
        self.p_values_return = torch.tensor([0.0, self.init_usd_amount], dtype=torch.float32, device=self.device)
        self.prev_action = torch.tensor([0],dtype=torch.int32, device=self.device)

    def _update_p_values(self):
        self.p_values_return[0] = self.p_values_return[1]
        self.p_values_return[1] = self.calcul_portfolio_value()

    def _buy(self):
        cost_fees = calcul_cost(self.total_amount[0], self.cost_rate)
        usd_price = self.total_amount[0] - cost_fees
        self.total_amount[1] = convert_to_btc(usd_price, self.btc_value)
        self.total_pnl[0] = self.total_amount[0]
        self.total_pnl[2] = cost_fees
        self.total_amount[0] = 0

    def _sell(self):
        usd_price = convert_to_usd(self.total_amount[1], self.btc_value)
        cost_fees = calcul_cost(usd_price, self.cost_rate)
        self.total_amount[0] = usd_price - cost_fees
        self.total_pnl[2] += cost_fees
        self.total_pnl[1] = usd_price - self.total_pnl[0]
        self.total_pnl[3] = profit_and_loss(self.total_pnl)
        self.total_amount[1] = 0

    def _all_reset(self, train:bool=True):
        if train:
            min_steps_left = 500
            max_macro_idx = self.size - (min_steps_left // 24) - 1
            random_day = torch.randint(0, max_macro_idx, (1,), device=self.device).item() if max_macro_idx > 0 else 0
            # Synchronised the Micro and Macro index
            micro_start = random_day * 24
            micro_end = micro_start + 23
            self.time = torch.tensor([random_day, micro_start, micro_end], dtype=torch.int32, device=self.device)
        else: self.time = torch.tensor([0, 0, 23])
        self.seq.fill_(0)
        # Reset Portfolio Value
        self.total_amount = torch.tensor([self.init_usd_amount, 0.0], dtype=torch.float32, device=self.device)
        self.p_values_return = torch.tensor([0.0, self.init_usd_amount], dtype=torch.float32, device=self.device)
        self.total_pnl.fill_(0.0)
        self.btc_value.fill_(0.0)
        self.prev_action.fill_(0)

    def _next(self):
        self.time[1] += 1
        self.time[2] += 1
        self.seq += 1
        if self.seq > 23:
            self.time[0] += 1
            self.seq = torch.tensor([0])

    def get_pnl(self) -> float: 
        pnl = self.total_pnl[3]
        if torch.isnan(pnl).any() or torch.isinf(pnl).any(): return 0.0
        return pnl.item()

    def calcul_portfolio_value(self) -> Tensor:
        return self.total_amount[0] if self.total_amount[0] > 0.0 else convert_to_usd(self.total_amount[1], self.btc_value)

    # Create a group state
    def new_state(self) -> tuple[Tensor, Tensor]:
        macro_idx = int(min(self.time[0].item(), self.size - 1))
        price_idx = int(min(self.time[2].item(), self.price.shape[0] - 1))
        start = self.time[1].item()
        end = self.time[2].item() + 1  
        daily_trades = self.hour_trade[start:end]
        macro_days = self.macro_trade[macro_idx]
        self.btc_value = self.price[price_idx]  # mise à jour du prix courant
        return daily_trades, macro_days

    # mask actions
    def get_action_mask(self) -> Tensor:
        mask = [True, True, True]
        if self.total_amount[1] < 1.0: mask[2] = False
        else: mask[1] = False
        return torch.tensor(mask, dtype=torch.bool, device=self.device).reshape(1,-1)

    # Reset the env to 0
    def reset(self, train:bool=True) -> tuple[Tensor, Tensor]:
        self._all_reset(train)
        macro_idx = int(self.time[0].item())
        price_idx = int(self.time[1].item())  # ou time[2] ? À voir selon ta logique
        self.btc_value = self.price[price_idx]
        daily_trades = self.hour_trade[self.time[1]:self.time[2]+1]  # +1 pour avoir 24h
        macro_days = self.macro_trade[macro_idx]
        return daily_trades, macro_days

    # The next step of env
    def step(self, action:Tensor) -> Any:
        future_idx = int(min(self.time[0].item() + 1, self.size - 1))
        state_pred = self.state_pred[future_idx] if self.state_pred is not None else None
        action_int = action.item()
        # Buy
        if action_int == 1: self._buy()
        # Sell
        elif action_int == 2:
            action_int = -1
            self._sell()
        self._next()
        next_state = self.new_state()
        self._update_p_values()
        return_ = return_log(self.p_values_return, self.device)
        #Compute the reward
        reward = reward_func(return_, action, self.prev_action, self.cost_rate)
        self.prev_action.fill_(action_int)
        done = self.time[2] == self.hour_trade.shape[0]
        truncate = self.calcul_portfolio_value() == 0
        return next_state, reward, state_pred, truncate, done
