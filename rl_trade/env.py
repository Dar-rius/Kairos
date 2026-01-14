import pandas as pd
import numpy as np
from .compute import return_log, calcul_cost, calcul_sharpe_ratio, profit_and_loss, reward_func
from .processing import convert_to_btc, convert_to_usd
import torch
from torch import Tensor
import gymnasium as gym

# ******* ENV **********
class Env(gym.Env):
    def __init__(self, hour_trade:pd.DataFrame, macro_trade:pd.DataFrame, price:pd.Series, state_pred:pd.Series=None, amount_usd:int=100000.0, cost_rate:float=0.001, device:str='cuda:0'):
        self.init_usd_amount = amount_usd
        # Total PnL [Buy Price, PnL Brut, Fees, PnL Final]
        self.total_pnl: list = [0.0, 0.0, 0.0, 0.0]
        self.btc_values: list[float] = []
        self.hour_trade = hour_trade.to_numpy()
        self.macro_trade = macro_trade.to_numpy()
        self.state_pred = state_pred.to_numpy() if state_pred is not None else None
        self.price = price.to_numpy()
        # Time for trades [Day, Start Hour, Last Hour]
        self.time = [0, 0, 23]
        self.total_amount: dict = {0: self.init_usd_amount, 1: 0.0}
        self.cost_rate = cost_rate
        self.size = self.macro_trade.shape[0]
        self.seq: int = 24
        self.observation_space = [self.hour_trade.shape[1], self.macro_trade.shape[1]]
        self.action_space = 3
        self.device = device
        self.p_values_return = [0.0, self.init_usd_amount]

    def _update_p_values(self):
        self.p_values_return[0] = self.p_values_return[1]
        self.p_values_return[1] = self.calcul_portfolio_value()

    def _buy(self):
        cost_fees = calcul_cost(self.total_amount[0], self.cost_rate)
        usd_price = self.total_amount[0] - cost_fees
        self.total_amount[1] = convert_to_btc(usd_price, self.btc_values[-1])
        self.total_pnl[0] = self.total_amount[0]
        self.total_pnl[2] = cost_fees
        self.total_amount[0] = 0

    def _sell(self):
        usd_price = convert_to_usd(self.total_amount[1], self.btc_values[-1])
        cost_fees = calcul_cost(usd_price, self.cost_rate)
        self.total_amount[0] =  usd_price - cost_fees
        self.total_pnl[2] += cost_fees
        self.total_pnl[1] = usd_price - self.total_pnl[0]
        self.total_pnl[3] = profit_and_loss(self.total_pnl)
        self.total_amount[1] = 0

    def _all_reset(self, train:bool=True):
        if train:
            min_steps_left = 500
            max_macro_idx = self.size - (min_steps_left // 24) - 1
            random_day = np.random.randint(0, max_macro_idx) if max_macro_idx > 0 else 0
            # Synchronised the Micro and Macro index
            micro_start = random_day * 24
            micro_end = micro_start + 23
            self.time = [random_day, micro_start, micro_end]
        else: self.time = [0, 0, 23]
        self.seq = 0
        # Reset Portfolio Value
        self.total_amount = {0: self.init_usd_amount, 1: 0.0}
        self.p_values_return = [self.init_usd_amount, 0.0]
        self.total_pnl = [0.0, 0.0, 0.0, 0.0] # Reset PnL
        self.btc_values = []

    def _next(self):
        self.time[1] += 1
        self.time[2] += 1
        self.seq += 1
        if self.seq > 23:
            self.time[0] += 1
            self.seq = 0

    def get_pnl(self): return self.total_pnl[3]

    def calcul_portfolio_value(self) -> float:
        return self.total_amount[0] if self.total_amount[0] > 0.0 else convert_to_usd(self.total_amount[1], self.btc_values[-1])

    # Create a group state
    def new_state(self):
        daily_trades = self.hour_trade[self.time[1]:self.time[2]]
        macro_days = self.macro_trade[self.time[0]]
        self.btc_values.append(self.price[self.time[2]])
        self._next()
        return [daily_trades, macro_days]

    def get_action_mask(self) -> Tensor:
        mask = [True, True, True]
        if self.total_amount[1] < 1.0: mask[2] = False
        else: mask[1] = False
        return torch.tensor(mask, dtype=torch.bool, device=self.device).reshape(1,-1)

    # Reset the env to 0
    def reset(self, train:bool=True):
        self._all_reset(train)
        return self.new_state()

    # The next step of env
    def step(self, action:int, entropy_b:float=None) -> tuple:
        state = self.new_state()
        future_idx = min(self.time[0] + 1, self.size - 1)
        state_pred = self.state_pred[future_idx] if self.state_pred is not None else None
        # Sell
        if action == 2: self._sell()
        # Buy
        elif action == 1: self._buy()
        self._update_p_values()
        return_ = 0
        if self.p_values_return[0] > 0.0 and self.p_values_return[1] > 0.0:
            return_ = return_log(self.p_values_return[1], self.p_values_return[0]) if len(self.btc_values) > 1 else 0
        done = True if self.time[2] == self.hour_trade.shape[0] else False
        truncate = True if self.calcul_portfolio_value() == 0 else False
        #Compute the reward
        reward = reward_func(return_, self.cost_rate, action, entropy_b)
        return (state, reward, state_pred, truncate, done)
