import pandas as pd
import numpy as np
from scipy.stats import entropy
import math
import torch
from torch import Tensor

BETA = .25

def reward_func(return_:Tensor, action:Tensor, entropy_b:Tensor|None, entropy_low: float = .3) -> float:
    if entropy_b is None: return 0.0
    action = torch.where(action == 2, -1.0, action.float())
    raw_return = return_ * 100
    second_macro = torch.clamp(entropy_b - entropy_low, min=0.0)
    macro_strat = torch.abs(action) * BETA *  second_macro
    final_reward = torch.tanh(raw_return - macro_strat)
    return  final_reward.item()

#Compute the sharpe ration
def calcul_sharpe_ratio(portfolio_value: list) -> float:
    if len(portfolio_value) < 2: return 0.0
    val_arr = np.array(portfolio_value)
    returns = np.diff(val_arr) / val_arr[:-1]
    std_dev = np.std(returns)
    if std_dev > 1e-8:
        sharpe = (np.mean(returns) / std_dev) * np.sqrt(365 * 24)
        return float(sharpe)
    else:
        return 0.0

def max_dd(portfolio: list[float]) -> Tensor:
    values = torch.tensor(portfolio)
    if values.shape[0] < 2: return torch.tensor(0.0)
    peak = torch.cummax(values, dim=0).values
    drawdowns = (peak - values) / (peak + 1e-9)
    mdd = torch.max(drawdowns)
    return mdd

def calcul_cost(amount: Tensor, cost_rate: Tensor, device:str) -> Tensor: return amount * cost_rate

#def compute_entropy(prob: list[float]): return entropy(prob, base=2)

def profit_and_loss(total_price:Tensor) -> Tensor: return  total_price[2] - total_price[3]

def return_log_vec(data: list, device:str) -> Tensor:
    data = torch.tensor(data, dtype=torch.float32, device=device)
    p_return = torch.log(data[1:]/data[:-1])
    return p_return

#Compute the return log
def return_log(data: Tensor, device:str) -> Tensor: 
    if data[1] > 1e-8 and data[0] > 1e-8: 
        return torch.log(data[1] / data[0])
    return torch.tensor(0.0, device=device)
