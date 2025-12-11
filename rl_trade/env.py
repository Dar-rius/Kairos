import pandas as pd
from .compute import return_log, calcul_sharpe_ratio, calcul_total_profit, reward
from .processing import df_to_list, concat_df, convert_tensor_to_list, convert_to_btc, convert_to_usd
from collections import namedtuple, deque

# Create a group sequence
class SequenceGroup:
    def __init__(self, memory_size: int):
        self.memory = deque([], maxlen=memory_size)
        #self.micro_day = namedtuple("MicroDay", ('open', 'high', 'low', 'close', 'volume'))
        #self.micro = namedtuple("Micro", ('data'))
        #self.macro = namedtuple("Macro", ('open', 'high', 'low', 'close', 'volume'))

    #def push_macro(self, *args): self.macro(*args)

    #def push_micro(self, *args): self.micro(self.micro_day(*args))

    def push(self, data: list(list)):
        self.memory.append(data)
        
    def clear(self):
        self.memory.clear()

    def sample(self): return self.memory 

    #Show the size of memory
    def len(self): return len(self.memory)

# ******* ENV **********

class Env:
    def __init__(self, daily_trade: pd.DataFrame, macro_trade: pd.DataFrame, memory_size: int, amount_usd: int = 100000.0, cost_rate: float = 0.001):
        self.portfolio_values: list[float] = []
        self.btc_values: list[float] = []
        #self.memory_size = memory_size
        self.historic_data = pd.DataFrame(data = {'action':list[int], 'portfolio': list[float]})
        self.reward: float = 0
        self.daily_trade = daily_trade
        self.macro_trade = macro_trade
        self.metric = pd.DataFrame(data={'date':[], "sharpe ratio": [], "tp": []})
        self.total_amount: dict = {0: amount_usd, 1: 0}
        #self.first_minute = 0
        #self.last_minute = memory_size
        self.sequence_group = SequenceGroup(memory_size)
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
    def create_batch(self, n_days:int = 85, n_minutes:int = 60) -> list: 
        #dataset = self.data.loc[self.first_minute : self.last_minute]
        #values = df_to_list(dataset)
        daily_trades = [self.daily_trade[i:i + n_minutes] for i in range(0, self.daily_trade.shape[0], n_minutes)]
        macro_trades = [self.macro_trade[i:i + n_days] for i in range(0, self.macro_trade.shape[0], n_days)]
        self.sequence_group.push([daily_trades, macro_trades])
        #self.btc_value = values[3][-1]
        #check_next_nb_time = len(self.data.loc[self.first_minute:, "Open"])
        
        #check the rest quantity of data
        #self.first_minute = self.last_minute
        #if  check_next_nb_time > self.MEMORY_SIZE:
            #self.last_minute += self.MEMORY_SIZE
        #else:
            #self.last_minute += check_next_nb_time 
            #done = True
        return  self.sequence_group.sample()[self.seq]

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

        #Compute the reward
        self.portfolio_values.append(self.calcul_portfolio_value())
        self.btc_values.append(self.btc_value)
        return_ = return_log(self.btc_values[-1], self.btc_values[-2])
        reward: float = reward(return_, self.cost_rate, action, prob)
        self.seq += 1
        state = self.sequence_group.sample()[self.seq]
        return (state, reward)
