import pandas as pd
from .compute import return_log, calcul_sharpe_ratio, calcul_total_profit, reward
from .processing import df_to_list, concat_df, convert_tensor_to_list, convert_to_btc, convert_to_usd
from collections import namedtuple, deque
#import cudf as cu
#import cupy as cp

# Create a group sequence
class SequenceGroup:
    def __init__(self, memory_size: int):
        self.memory = deque([], maxlen=memory_size)
        #self.micro_day = namedtuple("MicroDay", ('open', 'high', 'low', 'close', 'volume'))
        #self.micro = namedtuple("Micro", ('data'))
        #self.macro = namedtuple("Macro", ('open', 'high', 'low', 'close', 'volume'))

    #def push_macro(self, *args): self.macro(*args)

    #def push_micro(self, *args): self.micro(self.micro_day(*args))

    def push(self, data: list):
        self.memory.append(data)
        
    def clear(self):
        self.memory.clear()

    def sample(self): return self.memory 

    #Show the size of memory
    def len(self): return len(self.memory)

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
        self.start_batch: int = 0
        self.metric = pd.DataFrame(data={'date':[], "sharpe ratio": [], "tp": []})
        self.total_amount: dict = {0: amount_usd, 1: 0}
        self.btc_value: float = 0.0
        self.cost_rate = cost_rate
        self.seq: int = 0

    def __buy(self):
        print(self.btc_value)
        self.total_amount[1] = convert_to_btc(self.total_amount[0], self.btc_value)
        self.total_amount[0] = 0
        
    def __sell(self):
        self.total_amount[0] = convert_to_usd(self.total_amount[1], self.btc_value)
        self.total_amount[1] = 0

    def calcul_portfolio_value(self) -> float:
        if self.total_amount[0] > 0:
            return self.total_amount[0]
        else:
            return convert_to_usd(self.total_amount[1], self.btc_value)

    #Create a group state
    def create_batch(self) -> tuple: 
        days_trade = self.daily_trade.shape[0] // 1440
        daily_trades = self.daily_trade[self.start_batch:self.n_days].to_numpy()
        macro_days = self.macro_trade[self.start_batch:self.n_days].to_numpy()
        self.start_batch += self.n_days
        self.n_days += self.n_days
        print(days_trade)
        return (daily_trades, macro_days)

    #Reset the env to 0
    def reset(self): return self.create_batch()

    #The next step of env
    def step(self, action: int, prob: list[float]) -> tuple:
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

        state = self.create_batch()
        self.portfolio_values.append(self.calcul_portfolio_value())
        #the value [1] is just a test
        self.btc_values.append(state[0][1])
        return_ = return_log(self.btc_values[-1], self.btc_values[-2])
        #Compute the reward
        reward: float = reward(return_, self.cost_rate, action, prob)
        return (state, reward)
