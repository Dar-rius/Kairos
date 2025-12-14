import pandas as pd 
from rl_trade.env import Env, SequenceGroup, df_to_list 
import unittest
from pandas.testing import assert_frame_equal
import numpy as np
#import cudf as cu
#import cupy as cp

MEMORY_SIZE = 5
df = pd.read_csv("./data_off/unit_test/unit_test_df_price.csv")
jf = df.drop('Datetime_utc', axis=1)
df_1 = pd.read_csv("./data_off/unit_test/unit_test_df_metric.csv")
df_1 = df_1.drop(['date', 'state'], axis=1)
sequence_group = SequenceGroup(MEMORY_SIZE)

def test_reset():
    env = Env(daily_trade=df, macro_trade=df_1, n_days=10)
    tab1, tab2 = env.reset()
    daily_trades = df[0:10].to_numpy()
    macro_trades =  df_1[0:10].to_numpy()
    np.testing.assert_equal(tab1, daily_trades)
    np.testing.assert_equal(tab2, macro_trades)
    

def test_calcul_portfolio_value():
    env = Env(daily_trade=df, macro_trade=df_1, n_days=85)
    result = env.calcul_portfolio_value()
    assert result == 100000.0

"""
# Test all case of action
class TestStep:
    env = Env(daily_trade=df, macro_trade=df_1, n_days=85)
    env.reset()
    seq = SequenceGroup(MEMORY_SIZE)

    def test_buy(self):
        result = self.env.step(1, [0.0, 0.0, 0.0])
        dataset = df.loc[5 : 10]
        values = df_to_list(dataset)
        self.seq.push(values[0], values[1], values[2], values[3], values[4])
        assert result  ==  (self.seq.sample(), False, 0.0)

    def test_sell(self):
        result = self.env.step(-1, [0.0, 0.0, 0.0])
        dataset = df.loc[5 : 10]
        values = df_to_list(dataset)
        self.seq.clear()
        self.seq.push(values[0], values[1], values[2], values[3], values[4])
        assert result  ==  (self.seq.sample(), False, 0.0)
            
    def test_null(self):
        result = self.env.step(0, [0.0, 0.0, 0.0])
        dataset = df.loc[5 : 10]
        values = df_to_list(dataset)
        self.seq.clear()
        self.seq.push(values[0], values[1], values[2], values[3], values[4])
        assert result  ==  (self.seq.sample(), False, 0.0)
"""
