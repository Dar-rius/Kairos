import pandas as pd
from .compute import calcul_sharpe_ratio, calcul_total_profit
from .processing import to_df, convert_and_save_df, concat_df
from collections import namedtuple, deque

def convert_to_btc(amount_usd: float, btc_value: float): return amount_usd / btc_value

def convert_to_usd(amount_btc: float, btc_value: float): return amount_btc * btc_value

def add_data(dataset: pd.DataFrame) -> tuple:
    open_ = dataset["Open"].to_list()
    high_ = dataset["High"].to_list()
    low_ = dataset["Low"].to_list()
    close_ = dataset["Close"].to_list()
    volume_ = dataset["Volume"].to_list()
    return (open_, high_, low_, close_, volume_)


class SequenceGroup:
    def __init__(self, memory_size:int):
        self.memory = deque([], maxlen=memory_size)
        self.sequence = namedtuple("Sequence", ('open', 'high', 'low', 'close', 'volume'))

    def push(self, *args):
        self.memory.append(self.sequence(*args))
        
    def clear(self):
        self.memory.clear()

    def sample(self) -> list:
        return self.memory 

    def len(self) -> int:
        return len(self.memory)


class Env:
    def __init__(self, data: pd.DataFrame, amount_usd: int = 100000.0, memory_size: int = 5):
        self.portfolio_values: list[float] = []
        self.btc_values: list[float] = []
        self.MEMORY_SIZE = memory_size
        self.historic_data = pd.DataFrame(data = {'action':list[int], 'portfolio': list[float]})
        self.reward: float = 0
        self.data: pd.DataFrame = data
        self.metric = pd.DataFrame(data={'date':[], "sharpe ratio": [], "tp": []})
        self.total_amount: dict = {0: amount_usd, 1: 0}
        self.first_minute = 0
        self.last_minute = memory_size
        self.sequence_group = SequenceGroup(memory_size)
        self.btc_value: float = 0.0


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


    def create_batch(self) -> tuple[list, bool]: 
        if self.sequence_group.len() > 0:
            self.sequence_group.clear()

        done = False
        dataset = self.data.loc[self.first_minute : self.last_minute]
        values = add_data(dataset)
        self.sequence_group.push(values[0], values[1], values[2], values[3], values[4])
        print(values[3])
        self.btc_value = values[3][-1]
        check_next_nb_time = len(self.data.loc[self.first_minute:, "Open"])

        self.first_minute = self.last_minute
        if  check_next_nb_time > self.MEMORY_SIZE:
            self.last_minute += self.MEMORY_SIZE
        else:
            self.last_minute += check_next_nb_time 
            done = True

        return (self.sequence_group.sample(), done)


    def reset(self): return self.create_batch()


    def step(self, action: int) -> tuple:
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

        self.portfolio_values.append(self.calcul_portfolio_value())
        self.btc_values.append(self.btc_value)
        reward: float = calcul_sharpe_ratio(self.portfolio_values, self.btc_values)
        state, done = self.create_batch()
        return (state, done, reward)
