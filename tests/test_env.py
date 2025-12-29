import pandas as pd 
from rl_trade.env import Env 
from pandas.testing import assert_frame_equal
import numpy as np
import torch
import random

df = pd.read_csv("./data_off/unit_test/norm_price.csv").iloc[:, 1:]
df_1 = pd.read_csv("./data_off/unit_test/metric.csv").iloc[:, 1:]
price = pd.read_csv("./data_off/unit_test/price_close.csv")["Close"]
state_p = pd.read_csv("./data_off/unit_test/state.csv")["state"]
prob = torch.zeros([1], dtype=torch.float32)

def test_calcul_portfolio_value():
    env = Env(hour_trade=df, macro_trade=df_1, price=price)
    result = env.calcul_portfolio_value()
    assert result == 100000.0

# Test all case of action
class TestStep:
    env = Env(hour_trade=df, macro_trade=df_1, price=price, state_pred=state_p)
    env.reset()

    def test_buy(self):
        state, reward, _, _, done = self.env.step(1, prob)
        daily_trades = df[1:24].to_numpy()
        macro_trades =  df_1.loc[0].to_numpy()
        np.testing.assert_equal(daily_trades, state[0])
        np.testing.assert_equal(macro_trades,state[1])
        assert reward  !=  0.0
        assert self.env.total_pnl[0] > 0.0
        assert self.env.total_pnl[2] > 0.0
        assert not done

    def test_sell(self):
        state, reward, _, _, done = self.env.step(2, prob)
        daily_trades = df[2:25].to_numpy()
        macro_trades =  df_1.loc[0].to_numpy()
        np.testing.assert_equal(daily_trades, state[0])
        np.testing.assert_equal(macro_trades,state[1])
        assert reward  !=  0.0
        assert not done

    def test_null(self):
        state, reward, _, _, done = self.env.step(0, prob)
        daily_trades = df[3:26].to_numpy()
        macro_trades =  df_1.loc[0].to_numpy()
        np.testing.assert_equal(daily_trades, state[0])
        np.testing.assert_equal(macro_trades,state[1])
        assert reward  ==  -0.0
        assert not done
    
    # Test if we train an agent
    def test_finish_batch(self):
        start = 3
        end = 26
        start_n = 0
        for _ in range(0,48):
            start += 1
            end += 1
            start_n = start_n + 1 if self.env.seq == 0 else start_n
            daily_trades = df[start:end].to_numpy()
            macro_trades = df_1.loc[start_n].to_numpy()
            state_estim = state_p.loc[start_n]
            choice = random.choice([1, 2])
            state, _, state_pred, _, done = self.env.step(choice, prob)
            np.testing.assert_equal(daily_trades, state[0])
            np.testing.assert_equal(macro_trades, state[1])
            np.testing.assert_equal(state_pred, state_estim)
            if done: break

    def test_reset(self):
        self.env.reset()
        assert self.env.calcul_portfolio_value() == 100000.0
