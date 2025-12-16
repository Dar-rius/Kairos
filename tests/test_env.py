import pandas as pd 
from rl_trade.env import Env, df_to_list 
import unittest
from pandas.testing import assert_frame_equal
import numpy as np
#import cudf as cu
#import cupy as cp
import random
import torch

MEMORY_SIZE = 5
df = pd.read_csv("./data_off/unit_test/unit_test_df_price.csv")
jf = df.drop('Datetime_utc', axis=1)
df_1 = pd.read_csv("./data_off/unit_test/unit_test_df_metric.csv")
df_1 = df_1.drop(['date', 'state'], axis=1)
prob = torch.zeros([3], dtype=torch.float32)

def test_reset():
    env = Env(daily_trade=df, macro_trade=df_1, n_days=10)
    tab1, tab2 = env.reset()
    n_days = 10 * 1440
    daily_trades = df[0:n_days].to_numpy().reshape(10, 24, 60, -1)
    macro_trades =  df_1[0:10].to_numpy()
    np.testing.assert_equal(tab1, daily_trades[0][0])
    np.testing.assert_equal(tab2, macro_trades[0])
    

def test_calcul_portfolio_value():
    env = Env(daily_trade=df, macro_trade=df_1, n_days=85)
    result = env.calcul_portfolio_value()
    assert result == 100000.0

# Test all case of action
class TestStep:
    env = Env(daily_trade=df, macro_trade=df_1, n_days=10)
    env.reset()

    def test_buy(self):
        state, reward, done = self.env.step(1, prob)
        n_days = 10 * 1440
        daily_trades = df[0:n_days].to_numpy().reshape(10, 24, 60, -1)
        macro_trades =  df_1[0:10].to_numpy()
        np.testing.assert_equal(state[0], daily_trades[0][1])
        np.testing.assert_equal(state[1], macro_trades[0])
        assert reward  !=  0
        assert done == False

    def test_sell(self):
        state, reward, done = self.env.step(-1, prob)
        n_days = 10 * 1440
        daily_trades = df[0:n_days].to_numpy().reshape(10, 24, 60, -1)
        macro_trades =  df_1[0:10].to_numpy()
        np.testing.assert_equal(state[0], daily_trades[0][2])
        np.testing.assert_equal(state[1], macro_trades[0])
        assert reward  !=  0
        assert done == False
            
    def test_null(self):
        state, reward, done = self.env.step(1, prob)
        n_days = 10 * 1440
        daily_trades = df[0:n_days].to_numpy().reshape(10, 24, 60, -1)
        macro_trades =  df_1[0:10].to_numpy()
        np.testing.assert_equal(state[0], daily_trades[0][3])
        np.testing.assert_equal(state[1], macro_trades[0])
        assert reward  !=  0
        assert done == False
    """
    def test_finish_batch(self):
        done = False
        while not done:
            result = self.env.step(1, prob)  
    """        
