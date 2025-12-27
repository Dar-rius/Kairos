import pandas as pd
import numpy as np
from scipy.stats import entropy
import math
from torch import Tensor
from .processing import convert_tensor_to_list

BETA = 1

def reward_func(return_: float, cost_rate: float, action: int, entropy_b: Tensor, entropy_low: float = .3) -> float:
    entropy_b = entropy_b.item()
    # Compute the micro strategy for trading 1h
    first_micro = return_ * action
    second_micro = calcul_cost(abs(action), cost_rate)
    micro_strat = first_micro - second_micro
    # Compute the macro strategy for detect the state of market
    first_macro = BETA * abs(action)
    second_macro = max(0, entropy_b - entropy_low)
    macro_strat = first_macro * second_macro
    return micro_strat - macro_strat

#Compute the sharpe ration
def calcul_sharpe_ratio(data_p: list[float], data_btc: list[float]) -> float:
    if len(data_p) < 2:
        return 0.0
    n = len(data_p)
    #compute the return of portfolio
    portfolio_return = [return_log(data_p[i], data_p[i-1]) for i in range(1, n)]
    btc_return = [return_log(data_btc[i], data_btc[i-1]) for i in range(1, n)]
    #compute the return excess, mean and the derivating
    excess = [p - btc for p, btc in zip(portfolio_return, btc_return)]
    excess_avg = np.mean(excess)
    excess_std = np.std(excess)
    #compute the sharpe ratio
    sr = excess_avg/excess_std
    if np.isnan(sr):
        return 0.0
    return sr.item()

def calcul_cost(amount: float, cost_rate: float): return amount * cost_rate

def compute_entropy(prob: list[float]): return entropy(prob, base=2)

def profit_and_loss(total_price:list): return  total_price[2] - total_price[3]

#Compute the return log
def return_log(current: float, previous: float): return math.log(current / previous)
