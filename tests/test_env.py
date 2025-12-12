import pandas as pd 
from rl_trade.env import Env, SequenceGroup, df_to_list 
import unittest
from pandas.testing import assert_frame_equal

MEMORY_SIZE = 5
df = pd.read_csv("./data_off/unit_test/unit_test_df_price.csv")
df_1 = pd.read_csv("./data_off/unit_test/unit_test_df_metric.csv")
sequence_group = SequenceGroup(MEMORY_SIZE)

def test_reset():
    env = Env(daily_trade=df, macro_trade=df_1, memory_size=85)
    result = env.reset()
    daily_trades = []
    for _ in range(0, 9):
        hour_trade = []
        for _ in range(0,24):
            hour_trade = [df[i:i + 60] for i in range(0, df.shape[0], 60)]
        daily_trades.append(hour_trade)

    macro_trades = [df_1.loc[i] for i in range(0, df_1.shape[0])]
    sequence_group.push([daily_trades, macro_trades])
    print("test: ", sequence_group.sample()[0][1][0])
    assert len(result[0][0]) == len(sequence_group.sample()[0][0][0])
    for df_res, df_att in zip(result[0][0], sequence_group.sample()[0][0][0]):
        assert_frame_equal(df_res, df_att)
    for df_res, df_att in zip(result[1][1], sequence_group.sample()[0][1][1]):
        assert_frame_equal(df_res, df_att)

def test_calcul_portfolio_value():
    env = Env(daily_trade=df, macro_trade=df_1, memory_size=85)
    result = env.calcul_portfolio_value()
    assert result == 100000.0

# Test all case of action
class TestStep:
    env = Env(daily_trade=df, macro_trade=df_1, memory_size=85)
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
