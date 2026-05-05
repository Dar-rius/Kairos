import pandas as pd
import numpy as np
from .compute import return_log, calcul_cost, profit_and_loss, reward_func
from .processing import convert_to_btc, convert_to_usd
import torch
from torch import Tensor
import gymnasium as gym
from gymnasium import spaces
from typing import Any
from sklearn.preprocessing import StandardScaler
import joblib

# ******* ENV **********
class Env():
    def __init__(self, hour_trade:pd.DataFrame, macro_trade:pd.DataFrame,
                 price:pd.Series, state_pred:pd.Series=None, change_pred:pd.Series=None,
                 amount_usd:float=100000.0, cost_rate:float=0.001,
                 device:str='cpu', use_scaler:bool=False, dsr_eta:float=0.1, dsr_warmup:int=10
                 ):
        self.device = device
        self.init_usd_amount = amount_usd
        self.past_position = torch.tensor([0], dtype=torch.int, device=self.device)
        self.cash =  torch.tensor(self.init_usd_amount, dtype=torch.float32, device=self.device)
        self.btc_shorted = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.btc_held = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.btc_value = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        self.use_scaler = use_scaler
        if self.use_scaler:
            # Normalized all dataset
            self.micro_scaler = StandardScaler()
            self.macro_scaler = joblib.load("./agent/save/macro_scaler.pkl")
            scaled_hour = self.micro_scaler.fit_transform(hour_trade.values)
            scaled_macro = self.macro_scaler.transform(macro_trade.values)
            self.hour_trade = torch.tensor(scaled_hour, dtype=torch.float32, device=self.device)
            self.macro_trade = torch.tensor(scaled_macro, dtype=torch.float32, device=self.device)
        self.total_pnl = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.entry_price = torch.tensor(0.0, device=self.device)
        self.state_pred = torch.tensor(state_pred.values, dtype=torch.int8, device=self.device) if state_pred is not None else None
        self.change_pred = torch.tensor(change_pred.values, dtype=torch.int8, device=self.device) if change_pred is not None else None
        self.price = torch.tensor(price.values, dtype=torch.float32, device=self.device)
        # Time for trades [Day, Start Hour, Last Hour]
        self.time = torch.tensor([0, 0, 23], dtype=torch.int32, device=self.device)
        self.cost_rate = torch.tensor(cost_rate, device=self.device)
        self.size = self.macro_trade.shape[0]
        self.seq = torch.tensor(24, device=self.device)
        self.observation_space = self.hour_trade.shape[1], self.macro_trade.shape[1]
        self.action_space = 3
        self.p_values_return = torch.tensor([0.0, self.init_usd_amount], dtype=torch.float32, device=self.device)
        self.ema_a = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.ema_b = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.dsr_nu = torch.tensor(0.003, dtype=torch.float16, device=device)
        self.step_ = 0
        self.day_total = 0

    def _reset_dsr_stats(self):
        self.ema_a.fill_(0.0)
        self.ema_b.fill_(0.0)
        self.step_ = 0

    def _update_p_values(self):
        self.p_values_return[0] = self.p_values_return[1]
        self.p_values_return[1] = self.calcul_portfolio_value()

    def _buy(self):
        if self.btc_shorted > 0.0:
            #cover short
            needed_to_cover = self.btc_shorted * self.btc_value
            fees = calcul_cost(needed_to_cover, self.cost_rate)
            total_cost = needed_to_cover + fees
            if self.cash >= total_cost: 
                self.cash.sub_(total_cost)
                self.btc_shorted.fill_(0.0)
                self.entry_price.fill_(0.0)
                return
            else:
                can_buy = self.cash / (self.btc_value * (1 + self.cost_rate))
                fees = can_buy * self.btc_value * self.cost_rate
                pnl = (self.entry_price - self.btc_value) * can_buy
                self.total_pnl.add_(pnl - fees)

                self.btc_shorted.sub_(can_buy)
                self.cash.fill_(0.0)
        if self.cash > 0.0 and self.btc_shorted == 0.0:
            #buy BTC
            fees = calcul_cost(self.cash, self.cost_rate)
            balance = self.cash - fees
            self.btc_held.add_(balance / self.btc_value)
            self.entry_price.fill_(self.btc_value)
            self.cash.fill_(0.0)

    def _sell(self):
        p_btc = self.btc_held * self.btc_value
        cost_fees = calcul_cost(p_btc, self.cost_rate)
        pnl = (self.btc_value - self.entry_price) * self.btc_held
        self.total_pnl.add_(pnl - cost_fees)
        self.cash.add_(p_btc - cost_fees)
        self.btc_held.fill_(0.0)
        self.entry_price.fill_(0.0)

    def _short(self):
        if self.btc_held > 0.0:
            self._sell()
        p_t = self.calcul_portfolio_value()
        short_value = p_t * 0.5
        fees = calcul_cost(short_value, self.cost_rate)
        self.cash.add_(short_value - fees)
        self.btc_shorted.add_(short_value / self.btc_value)
        self.entry_price.fill_(self.btc_value)

    def _rebalance(self, target_pos: int):
        """
        target_pos: -1 (Short), 0 (Neutral), 1 (Long)
        """
        current_pos = 0
        if self.btc_held > 1e-8: current_pos = 1
        elif self.btc_shorted > 1e-8: current_pos = -1

        # Do nothing (HOLD)
        if target_pos == current_pos:
            return

        # Close the position if we was Long
        if current_pos == 1:
            p_btc = self.btc_held * self.btc_value
            fees = p_btc * self.cost_rate
            pnl = (self.btc_value - self.entry_price) * self.btc_held
            self.total_pnl.add_(pnl - fees)
            self.cash.add_(p_btc - fees)
            self.btc_held.fill_(0.0)
        
        # Cover the position if we was short
        elif current_pos == -1:
            needed_to_cover = self.btc_shorted * self.btc_value
            fees = needed_to_cover * self.cost_rate
            pnl = (self.entry_price - self.btc_value) * self.btc_shorted
            self.total_pnl.add_(pnl - fees)
            self.cash.sub_(needed_to_cover + fees)
            self.btc_shorted.fill_(0.0)

        self.entry_price.fill_(0.0)
        p_t = self.calcul_portfolio_value()
        
        if target_pos == 1: # On veut devenir LONG
            amount_to_use = self.cash * 0.95 
            fees = amount_to_use * self.cost_rate
            self.btc_held.fill_((amount_to_use - fees) / self.btc_value)
            self.cash.sub_(amount_to_use)
            self.entry_price.fill_(self.btc_value)
        
        #Go short market for 50% from Portfolio
        elif target_pos == -1: 
            short_value_usd = p_t * 0.5
            fees = short_value_usd * self.cost_rate
            self.cash.add_(short_value_usd - fees)
            self.btc_shorted.fill_(short_value_usd / self.btc_value)
            self.entry_price.fill_(self.btc_value)

    def liquidation(self) -> bool:
        p_t = self.calcul_portfolio_value()
        val_warn = self.init_usd_amount * 0.1
        return p_t.item() < val_warn

    def _all_reset(self, train:bool=True):
        if train:
            min_steps_left = 500
            max_macro_idx = self.size - (min_steps_left // 24) - 1
            random_day = torch.randint(0, max_macro_idx, (1,), device=self.device).item() if max_macro_idx > 0 else 0
            # Synchronised the Micro and Macro index
            micro_start = random_day * 24
            micro_end = micro_start + 23
            self.time = torch.tensor([random_day, micro_start, micro_end], dtype=torch.int32, device=self.device)
            self.day_total = self.size - int(self.time[0].item())
        else:
            self.time = torch.tensor([0, 0, 23], dtype=torch.int32, device=self.device)
            self.day_total = self.size
        print(self.day_total)
        self.seq.fill_(0)
        # Reset Portfolio Value
        self.p_values_return = torch.tensor([0.0, self.init_usd_amount], dtype=torch.float32, device=self.device)
        self.cash.fill_(self.init_usd_amount)
        self.total_pnl.fill_(0.0)
        self.btc_held.fill_(0.0)
        self.btc_shorted.fill_(0.0)
        self._reset_dsr_stats()

    def _next(self):
        self.time[1] += 1
        self.time[2] += 1
        self.seq += 1
        if self.seq > 23:
            self.time[0] += 1
            self.seq = torch.tensor([0])

    def get_pnl(self) -> float:
        pnl = self.total_pnl
        if torch.isnan(pnl).any() or torch.isinf(pnl).any(): return 0.0
        return pnl.item()

    def calcul_portfolio_value(self) -> Tensor:
        return self.cash + (self.btc_held * self.btc_value) - (self.btc_shorted * self.btc_value)

    # Create a group state
    def new_state(self) -> tuple[Tensor, Tensor, Tensor]:
        macro_idx = int(min(self.time[0].item(), self.size - 1))
        price_idx = int(min(self.time[2].item(), self.price.shape[0] - 1))
        start = self.time[1].item()
        end = self.time[2].item() + 1
        daily_trades = self.hour_trade[start:end]
        macro_days = self.macro_trade[macro_idx]
        self.btc_value = self.price[price_idx]
        current_pos = torch.tensor([self.get_current_position_type()],
                                   dtype=torch.float32, device=self.device)
        return daily_trades, macro_days, current_pos.clone()

    # mask actions
    def get_action_mask(self) -> Tensor:
        mask = [True, True, True]
        return torch.tensor(mask, dtype=torch.bool, device=self.device).reshape(1, -1)

    def get_current_position_type(self):
        if self.btc_held > 1e-8: return 1.0    # Long
        if self.btc_shorted > 1e-8: return -1.0 # Short
        return 0.0 # Cash

    # Reset the env to 0
    def reset(self, train:bool=True) -> tuple[Tensor, Tensor, Tensor]:
        self._all_reset(train)
        return self.new_state()

    # The next step of env
    def step(self, action:Tensor) -> Any:
        future_idx = int(min(self.time[0].item() + 1, self.size - 1))
        state_pred = self.state_pred[future_idx] if self.state_pred is not None else None
        change_pred = self.change_pred[future_idx] if self.change_pred is not None else None
        target_pos = int(action.item()) - 1 
        self._rebalance(target_pos)
        self._next()
        next_state = self.new_state()
        self._update_p_values()
        return_ = return_log(self.p_values_return, self.device)
        is_liquidated = self.liquidation()
        truncate = False
        reward = 0.0
        if is_liquidated:
            reward = -10.0
            truncate = True
        else:
            #Compute the reward
            self.step_+=1
            reward, self.ema_a, self.ema_b = reward_func(return_, self.step_, self.dsr_nu, self.ema_a, self.ema_b)
        done = self.time[2] == self.hour_trade.shape[0]
        return next_state, reward, state_pred, change_pred, truncate, done
