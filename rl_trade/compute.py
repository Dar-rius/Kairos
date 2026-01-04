import pandas as pd
import numpy as np
from scipy.stats import entropy
import math
from torch import Tensor
from .processing import convert_tensor_to_list

BETA = 0.1

def reward_func(return_: float, cost_rate: float, action: int, entropy_b: Tensor, entropy_low: float = .9) -> float:
    if entropy_b is None: return 0.0
    if action == 2: action = -1
    reward = return_ * 100.0
    if action != 0: reward -= (BETA * entropy_b)
    return np.clip(reward, -10.0, 10.0)

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
