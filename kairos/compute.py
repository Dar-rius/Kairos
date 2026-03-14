import pandas as pd
import numpy as np
from scipy.stats import entropy
import math
import torch
from torch import Tensor
from collections import deque

def reward_func(return_:Tensor, action: Tensor, prev_action:Tensor, fees:Tensor) -> float:
    if torch.isnan(return_).any() or torch.isinf(return_).any(): return -10.0
    if action == 2: action = torch.tensor([-1], device=action.device)
    gain = (action * return_) * 100
    cost = fees * 100 * torch.abs(action - prev_action)
    reward = gain - cost
    final_reward = torch.clamp(reward, -10.0, 10.0)
    return final_reward.item()

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
