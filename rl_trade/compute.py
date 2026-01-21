import pandas as pd
import numpy as np
from scipy.stats import entropy
import math
import torch
from torch import Tensor

BETA = .25

def reward_func(return_: Tensor, action: Tensor, entropy_b: Tensor, entropy_low: float = .3) -> float:
    if entropy_b is None: return 0.0
    action = torch.where(action == 2, -1.0, action.float())
    # Compute the macro strategy for detect the state of market
    second_macro = torch.clamp(entropy_b - entropy_low, min=0.0)
    macro_strat = torch.abs(action) * BETA *  second_macro
    return torch.clamp((return_ * 100) - macro_strat, -10.0, 10.0)

#Compute the sharpe ration
def calcul_sharpe_ratio(data_p: list[float], data_btc: list[float]) -> float:
    if len(data_p) < 2: return 0.0
    #compute the return of portfolio
    portfolio_return = return_log_vec(data_p)
    btc_return = return_log_vec(data_btc)
    #compute the return excess, mean and the derivating
    excess = portfolio_return - btc_return
    excess_avg = torch.mean(excess)
    excess_std = torch.std(excess)
    #compute the sharpe ratio
    sr = excess_avg/(excess_std + 1e-8)
    return torch.nan_to_num(sr, nan=0.0)

def max_dd(portfolio: list[float]) -> float:
    values = torch.tensor(portfolio)
    if values.shape[0] < 2: return 0.0
    peak = torch.cummax(values, dim=0).values
    drawdowns = (peak - values) / (peak + 1e-9)
    mdd = torch.max(drawdowns)
    return mdd


def calcul_trade_metrics(returns: list[float]) -> float:
    data = torch.tensor(returns)
    num_trades = data.shape[0]
    if  num_trades == 0: return 0.0
    gains = data[data > 0]
    loss = data[data < 0]
    win_rate = gains.shape[0] / num_trades
    loss_rate = loss.shape[0] / num_trades
    avg_win = torch.mean(gains) if gains.shape[0] > 0 else 0.0
    avg_loss = torch.abs(torch.mean(loss)) if loss.shape[0] > 0 else 0.0
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    return expectancy

def calcul_cost(amount: Tensor, cost_rate: Tensor, device:str): return amount * cost_rate

def compute_entropy(prob: list[float]): return entropy(prob, base=2)

def profit_and_loss(total_price:list): return  total_price[2] - total_price[3]

def return_log_vec(data: list) -> Tensor:
    data = torch.tensor(data, dtype=torch.float32)
    p_return = torch.log(data[1:]/data[:-1])
    return p_return

#Compute the return log
def return_log(data: Tensor, device:str) -> Tensor: 
    if data[1] > 1e-8 and data[0] > 1e-8: 
        return torch.log(data[1] / data[0])
    return torch.tensor(0.0, device=device)
