import pandas as pd
from .compute import return_log, calcul_sharpe_ratio, calcul_total_profit, reward_func
from .processing import df_to_list, concat_df, convert_to_btc, convert_to_usd
from collections import namedtuple, deque
from torch import Tensor
#import cudf as cu
#import cupy as cp

# ******* ENV **********

class Env:
    def __init__(self, daily_trade: pd.DataFrame, macro_trade: pd.DataFrame, n_days:int, amount_usd: int = 100000.0, cost_rate: float = 0.001):
        self.portfolio_values: list[float] = []
        self.btc_values: list[float] = []
        self.historic_data = pd.DataFrame(data = {'action':list[int], 'portfolio': list[float]})
        self.reward: float = 0
        self.daily_trade = daily_trade
        self.macro_trade = macro_trade
        self.n_days = n_days
        self.start_min: int = 0
        self.start_day: int = 0
        self.metric = pd.DataFrame(data={'date':[], "sharpe ratio": [], "tp": []})
        self.total_amount: dict = {0: amount_usd, 1: 0}
        self.cost_rate = cost_rate
        self.batch: list(type) = []
        self.initial_n_days = n_days
        self.time: list = [0,0]
        self.size = self.macro_trade.shape[0]

    def __buy(self):
        self.total_amount[1] = convert_to_btc(self.total_amount[0], self.btc_values[-1])
        self.total_amount[0] = 0
        
    def __sell(self):
        self.total_amount[0] = convert_to_usd(self.total_amount[1], self.btc_values[-1])
        self.total_amount[1] = 0

    def calcul_portfolio_value(self) -> float:
        return self.total_amount[0] if self.total_amount[0] > 0 else convert_to_usd(self.total_amount[1], self.btc_values[-1])

    #Create a group state
    def create_batch(self): 
        self.time =[0, 0] if self.time[0] == 10 else self.time
        n_minutes = self.n_days * 1440
        daily_trades = self.daily_trade[self.start_min:n_minutes].to_numpy().reshape(self.n_days, 24, 60, -1)
        macro_days = self.macro_trade[self.start_day:self.n_days].to_numpy()
        self.start_min = n_minutes
        self.start_day = self.n_days
        self.n_days += self.initial_n_days if self.n_days + self.initial_n_days < self.size else (self.n_days + self.initial_n_days) - self.size
        self.batch = [daily_trades, macro_days]

    def __all_reset(self):
        self.n_days = self.initial_n_days
        self.start_min = 0
        self.start_day = 0
        self.batch.clear()

    def __select_state(self) -> (list, bool):
        data = [self.batch[0][self.time[0]][self.time[1]], self.batch[1][self.time[0]]]
        self.time[0] = self.time[0] + 1 if self.time[1] == 23 else self.time[0]
        self.time[1] = 0 if self.time[1] == 23 else self.time[1] + 1
        done = True if self.time[0] == self.initial_n_days else False
        return (data, done)

    #Reset the env to 0
    def reset(self):
        if self.n_days > self.initial_n_days: self.__all_reset()
        self.create_batch()
        return self.__select_state()

    #The next step of env
    def step(self, action: int, prob: Tensor) -> tuple:
        state, done = self.__select_state()
        if done: return ([], 0.0, done)
        self.btc_values.append(state[0][-1][1])
        self.portfolio_values.append(self.calcul_portfolio_value())
        if action  == -1:
            trade_info = [action, self.calcul_portfolio_value()]
            self.historic_data = concat_df(self.historic_data,  trade_info)
            self.__sell()
        elif action == 1:
            trade_info = [action, self.calcul_portfolio_value()]
            self.historic_data = concat_df(self.historic_data,  trade_info)
            self.__buy()
        else:
            trade_info = [action, self.calcul_portfolio_value()]
            self.historic_data = concat_df(self.historic_data,  trade_info)
        return_ = return_log(self.btc_values[-1], self.btc_values[-2]) if len(self.btc_values) > 1 else 0
        #Compute the reward
        reward: float = reward_func(return_, self.cost_rate, action, prob)
        return (state, reward, done)
