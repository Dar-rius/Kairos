import pandas as pd
import numpy as np
import math
import torch
from torch import Tensor
from collections import deque

def reward_func(return_:Tensor, belief_probs: Tensor, position: int, prev_position: int, step:int, dsr_nu:Tensor, ema_a:Tensor, ema_b:Tensor, alpha: Tensor, gamma: Tensor, beta: Tensor) -> tuple[float, Tensor, Tensor]:
    if torch.isnan(return_).any() or torch.isinf(return_).any(): 
        return -5.0, ema_a, ema_b
    #is_flat = (torch.abs(return_) < 1e-7)
    # Compute delta A and B
    delta_a =  return_ - ema_a
    delta_b =  (return_**2) - ema_b
    # Compute A and B
    new_ema_a = ema_a + (dsr_nu * delta_a)
    new_ema_b = ema_b + (dsr_nu * delta_b)
    # if step is less than 4
    if step < 10: return torch.clamp(return_*50, -5.0, 5.0).item(), new_ema_a, new_ema_b
    #if is_flat.item(): return 0.0, new_ema_a, new_ema_b
    # Calcul du DSR
    epsilon = 1e-4
    variance = ema_b - (ema_a ** 2)
    # Calcul DSR
    numerator = (ema_b * delta_a) - (0.5 * ema_a * delta_b)
    denominator = torch.pow(torch.clamp(variance, min=0.0) + epsilon, 1.5)
    dsr = numerator / denominator
    # Compute the reward
    #belief_entropy = -torch.sum(belief_probs * torch.log(belief_probs + 1e-8), dim=-1)
    #max_entropy = torch.log(torch.tensor(3.0))
    #confidence = 1.0 - (belief_entropy / max_entropy)
    #stability = dsr * (1.0 + gamma * confidence)
    #penality = alpha * abs(position - prev_position)
    #total_reward = stability - penality
    #reward = torch.clamp(total_reward, -5.0, 5.0).item()
    reward = torch.clamp(dsr * 10, -5.0, 5.0).item()
    #print(f"Step: {step} | Return brut: {return_.item()} | DSR: {dsr} | Confidence: {confidence}| Total brute: {reward} ")
    return reward, new_ema_a, new_ema_b

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
