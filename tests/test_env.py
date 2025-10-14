import pandas as pd 
from rl_trade.env import Env, SequenceGroup
import unittest

MEMORY_SIZE = 5
df = pd.read_csv("btc-usd_dataset.csv")
env = Env(data = df)
sequence_group = SequenceGroup(MEMORY_SIZE)


def add_data(dataset: pd.DataFrame) -> tuple:
    open_ = dataset["Open"].to_list()
    high_ = dataset["High"].to_list()
    low_ = dataset["Low"].to_list()
    close_ = dataset["Close"].to_list()
    volume_ = dataset["Volume"].to_list()
    return (open_, high_, low_, close_, volume_)

    

def test_reset():
    result = env.reset()
    dataset =df.loc[0 : 5]
    values = add_data(dataset)
    sequence_group.push(values[0], values[1], values[2], values[3], values[4])

    assert result == (sequence_group.sample(), False) 

def test_calcul_portfolio_value():
    result = env.calcul_portfolio_value()
    assert result == 100000.0

def test_step():
    result_buy = env.step(1)
    result_sell = env.step(-1)
    result_none = env.step(0)

    dataset_1 = df.loc[5 : 10]
    values_1 = add_data(dataset_1)
    dataset_2 = df.loc[10 : 15]
    values_2 = add_data(dataset_2)
    dataset_3 = df.loc[15 : 20]
    values_3 = add_data(dataset_3)
    
   # assert result_buy ==


