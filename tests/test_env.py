import pandas as pd 
from rl_trade.env import Env, SequenceGroup, df_to_list 
import unittest

MEMORY_SIZE = 5
df = pd.read_csv("btc-usd_dataset.csv")
sequence_group = SequenceGroup(MEMORY_SIZE)

def test_reset():
    env = Env(data = df)
    result = env.reset()
    dataset = df.loc[0 : 5]
    values = df_to_list(dataset)
    sequence_group.push(values[0], values[1], values[2], values[3], values[4])
    assert result == (sequence_group.sample(), False) 

def test_calcul_portfolio_value():
    env = Env(data = df)
    result = env.calcul_portfolio_value()
    assert result == 100000.0

# Test all case of action
class TestStep:
    env = Env(data=df)
    env.reset()
    seq = SequenceGroup(MEMORY_SIZE)

    def test_buy(self):
        result = self.env.step(1)
        dataset = df.loc[5 : 10]
        values = df_to_list(dataset)
        self.seq.push(values[0], values[1], values[2], values[3], values[4])
        assert result  ==  (self.seq.sample(), False, 0.0)

    def test_sell(self):
        result = self.env.step(-1)
        dataset = df.loc[5 : 10]
        values = df_to_list(dataset)
        self.seq.clear()
        self.seq.push(values[0], values[1], values[2], values[3], values[4])
        assert result  ==  (self.seq.sample(), False, 0.0)
            
    def test_null(self):
        result = self.env.step(0)
        dataset = df.loc[5 : 10]
        values = df_to_list(dataset)
        self.seq.clear()
        self.seq.push(values[0], values[1], values[2], values[3], values[4])
        assert result  ==  (self.seq.sample(), False, 0.0)
