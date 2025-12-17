import pandas as pd
from .compute import return_log, calcul_sharpe_ratio, calcul_total_profit, reward_func
from .processing import df_to_list, concat_df, convert_to_btc, convert_to_usd
from collections import namedtuple, deque
from torch import Tensor
#import cudf as cu
#import cupy as cp

# ******* ENV **********

class Env:
    def __init__(self, daily_trade: pd.DataFrame, macro_trade: pd.DataFrame, price: pd.Series, n_days:int, amount_usd: int = 100000.0, cost_rate: float = 0.001):
        self.portfolio_values: list[float] = []
        self.btc_values: list[float] = []
        self.historic_data = pd.DataFrame(data = {'action':list[int], 'portfolio': list[float]})
        self.reward: float = 0
        self.hour_trade = daily_trade
        self.macro_trade = macro_trade
        self.time = [0, 0, 23]
        self.metric = pd.DataFrame(data={'date':[], "sharpe ratio": [], "tp": []})
        self.total_amount: dict = {0: amount_usd, 1: 0.0}
        self.cost_rate = cost_rate
        self.size = self.macro_trade.shape[0]
        self.price = price
        self.seq: int = 24

    def __buy(self):
        self.total_amount[1] = convert_to_btc(self.total_amount[0], self.btc_values[-1])
        self.total_amount[0] = 0
        
    def __sell(self):
        self.total_amount[0] = convert_to_usd(self.total_amount[1], self.btc_values[-1])
        self.total_amount[1] = 0

    def __all_reset(self):
        self.time = [1, 0, 23]
        self.seq = 0

    def __next(self):
        self.time[1] += 1
        self.time[2] += 1
        self.seq += 1
        if self.seq > 23:
            self.time[0] += 1
            self.seq = 0

    def calcul_portfolio_value(self) -> float:
        return self.total_amount[0] if self.total_amount[0] > 0.0 else convert_to_usd(self.total_amount[1], self.btc_values[-1])

    #Create a group state
    def new_state(self): 
        print(self.time[2])
        daily_trades = self.hour_trade[self.time[1]:self.time[2]].to_numpy()
        macro_days = self.macro_trade.loc[self.time[0]].to_numpy()
        self.btc_values.append(self.price.loc[self.time[2]]["Close"])
        self.__next()
        return [daily_trades, macro_days]

    #Reset the env to 0
    def reset(self):
        self.__all_reset()
        return self.new_state()

    #The next step of env
    def step(self, action: int, prob: Tensor) -> tuple:
        state = self.new_state()
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
        done = True if self.time[2] == self.hour_trade.shape[0] else False
        #Compute the reward
        reward: float = reward_func(return_, self.cost_rate, action, prob)
        return (state, reward, done)
