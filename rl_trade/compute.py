import pandas as pd
import numpy as np
from scipy.stats import entropy

#Compute the sharpe ration
def calcul_sharpe_ratio(data_p: list[float], data_btc: list[float]) -> float:
    if len(data_p) < 2:
        return 0.0
    n = len(data_p)

    #compute the return of portfolio
    portfolio_return = [pct_change(data_p[i], data_p[i-1]) for i in range(1, n)]
    btc_return = [pct_change(data_btc[i], data_btc[i-1]) for i in range(1, n)]

    #compute the return excess, mean and the derivating
    excess = [p - btc for p, btc in zip(portfolio_return, btc_return)]
    excess_avg = np.mean(excess)
    excess_std = np.std(excess)
    
    #compute the sharpe ratio
    sr = excess_avg/excess_std
    if np.isnan(sr):
        return 0.0

    return sr.item()

def compute_belief(data_p: list[float], cost: float, prob: list[float], beta: float) -> float:
    if len(data_p) < 2:
        return 0.0
    n = len(data_p)

    portfolio_return = pct_change(data_p[n-1], data_p[n-2])
    entropy_prob = entropy(prob, base=2)
    return portfolio_return - cost - (beta * entropy_prob)

def calcul_total_profit(precedent_tp: float, now_tp: float): return precedent_tp - now_tp

#Compute the return
def pct_change(current: float, previous: float): return (current - previous) / previous
