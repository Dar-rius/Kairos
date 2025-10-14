import pandas as pd
from statistics import *
import numpy as np

#Function to compute the sharpe ration in any timestamp
def calcul_sharpe_ratio(data_p: list[float], data_btc: list[float]) -> float:
    if len(data_p) != len(data_btc):
        raise ValueError("The size of portfolio data is not equal to btc data")

    n = len(data_p)
    #compute the return of portfolio
    portfolio_return = [pct_change(data_p[i], data_p[i-1]) for i in range(1, n)]
    btc_return = [pct_change(data_btc[i], data_btc[i-1]) for i in range(1, n)]

    #compute the return excess, mean et derivated
    excess = [p - btc for p, btc in zip(portfolio_return, btc_return)]
    excess_avg = np.mean(excess)
    excess_std = np.std(excess)

    #compute the sharpe ratio
    sr = excess_avg/excess_std
    return sr


#Function to compute the value of portfilio between t-1 and t
def calcul_total_profit(precedent_tp: float, now_tp: float): return precedent_tp - now_tp

def pct_change(previous: float, current : float): return (current - previous) / previous
