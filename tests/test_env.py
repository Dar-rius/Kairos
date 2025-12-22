import pandas as pd 
from rl_trade.env import Env, df_to_list 
import unittest
from pandas.testing import assert_frame_equal
import numpy as np
#import cudf as cu
#import cupy as cp
import random
import torch

df = pd.read_csv("./data_off/unit_test/unit_test_norm_price.csv")
df_1 = pd.read_csv("./data_off/unit_test/unit_test_df_metric.csv")
price = pd.read_csv("./data_off/unit_test/unit_test_price_close.csv")
prob = torch.zeros([1], dtype=torch.float32)

def test_reset():
    env = Env(daily_trade=df, macro_trade=df_1, price=price, n_days=10)
    tab1, tab2 = env.reset()
    daily_trades = df[0:23].to_numpy()
    macro_trades =  df_1.loc[1].to_numpy()
    np.testing.assert_equal(daily_trades, tab1)
    np.testing.assert_equal(macro_trades, tab2)

def test_calcul_portfolio_value():
    env = Env(daily_trade=df, macro_trade=df_1, price=price, n_days=85)
    result = env.calcul_portfolio_value()
    assert result == 100000.0

# Test all case of action
class TestStep:
    env = Env(daily_trade=df, macro_trade=df_1, price=price, n_days=10)
    env.reset()

    def test_buy(self):
        state, reward, done = self.env.step(1, prob)
        daily_trades = df[1:24].to_numpy()
        macro_trades =  df_1.loc[1].to_numpy()
        np.testing.assert_equal(daily_trades, state[0])
        np.testing.assert_equal(macro_trades,state[1])
        assert reward  !=  0.0
        assert not done

    def test_sell(self):
        state, reward, done = self.env.step(-1, prob)
        daily_trades = df[2:25].to_numpy()
        macro_trades =  df_1.loc[1].to_numpy()
        np.testing.assert_equal(daily_trades, state[0])
        np.testing.assert_equal(macro_trades,state[1])
        assert reward  !=  0.0
        assert not done
            
    def test_null(self):
        state, reward, done = self.env.step(0, prob)
        daily_trades = df[3:26].to_numpy()
        macro_trades =  df_1.loc[1].to_numpy()
        np.testing.assert_equal(daily_trades, state[0])
        np.testing.assert_equal(macro_trades,state[1])
        assert reward  ==  -0.0
        assert not done
    
    # Test if we train an agent
    def test_finish_batch(self):
        start = 3
        end = 26
        start_n = 1
        for _ in range(0,48):
            start += 1
            end += 1
            start_n = start_n + 1 if self.env.seq == 0 else start_n
            daily_trades = df[start:end].to_numpy()
            macro_trades =  df_1.loc[start_n].to_numpy()
            state, _, done = self.env.step(1, prob)
            np.testing.assert_equal(daily_trades, state[0])
            np.testing.assert_equal(macro_trades, state[1])
            if done: break
