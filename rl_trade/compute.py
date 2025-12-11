import pandas as pd
import numpy as np
from scipy.stats import entropy
import math

BETA = 1

def reward(return_: float, cost_rate: float, action: int, prob: list[float], entropy_low: float = .3) -> float:
    # Compute the micro strategy for trading 1h
    first_micro = return_ * action 
    second_micro = cost_rate * abs(action)
    micro_strat = first_micro - second_micro
    
    # Compute the macro strategy for detect the state of market
    first_macro = BETA * abs(action)
    second_macro = max(0, compute_entropy(prob) - entropy_low)
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

def compute_cost(amount: float, cost_rate: float): return amount * cost_rate

def compute_entropy(prob: list[float]): return entropy(prob, base=2)

def calcul_total_profit(precedent_tp: float, now_tp: float): return precedent_tp - now_tp

#Compute the return log
def return_log(current: float, previous: float): return math.log(current / previous)
