import pandas as pd
import numpy as np
from scipy.stats import entropy
import math
import torch
from torch import Tensor
from collections import deque
from math import *

def reward_func(return_:Tensor, mu: float, sigma2:float, sharpe_prev: float, eta: float=0.01) -> tuple[float, float, float, float]:
    if torch.isnan(return_).any() or torch.isinf(return_).any():
        return -10.0, mu, sigma2, sharpe_prev
    r = return_.item()
    delta = r - mu
    mu += eta * delta
    sigma2 += eta * (delta**2 - sigma2)
    sigma = sqrt(sigma2 + 1e-8)
    sharpe = mu / sigma
    reward_final = (sharpe - sharpe_prev) * 100
    sharpe_prev = sharpe
    return np.clip(reward_final, -50, 50).item(), mu, sigma2, sharpe_prev

#Compute the sharpe ration
def calcul_sharpe_ratio(portfolio_value: list) -> float:
    if len(portfolio_value) < 2: return 0.0
    val_arr = np.array(portfolio_value, dtype=np.float64)
    if not np.isfinite(val_arr).all(): return 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = np.diff(val_arr) / (val_arr[:-1] + 1e-9)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    std_dev = np.std(returns)
    if std_dev > 1e-8:
        sharpe = (np.mean(returns) / std_dev) * np.sqrt(365 * 24)
        if np.isnan(sharpe) or np.isinf(sharpe): return 0.0
        return float(sharpe)
    return 0.0

def max_dd(portfolio: deque[float], dd:bool=False) -> float:
    values = torch.tensor(list(portfolio), dtype=torch.float32)
    if values.shape[0] < 2: return 0.0
    if not torch.isfinite(values).all(): return 1.0
    if dd:
        peak = torch.max(values)
        drawdowns = (peak - values[-1]) / (peak + 1e-9)
        return drawdowns.item()
    peak = torch.cummax(values, dim=0).values
    drawdowns = (peak - values) / (peak + 1e-9)
    mdd = torch.max(drawdowns)
    return mdd.item()

def calcul_cost(amount: Tensor, cost_rate: Tensor) -> Tensor: return amount * cost_rate

#def compute_entropy(prob: list[float]): return entropy(prob, base=2)

def profit_and_loss(total_price:Tensor) -> Tensor: return  total_price[2] - total_price[3]

def return_log_vec(data: list, device:str) -> Tensor:
    data_ = torch.tensor(data, dtype=torch.float32, device=device)
    p_return = torch.log(data_[1:]/data_[:-1])
    return p_return

#Compute the return log
def return_log(data: Tensor, device:str) -> Tensor:
    if data[1] > 1e-8 and data[0] > 1e-8:
        return torch.log(data[1] / data[0])
    return torch.tensor(0.0, device=device)
