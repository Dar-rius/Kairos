import pandas as pd
import numpy as np
from scipy.stats import entropy
import math
import torch
from torch import Tensor

BETA = .2

def reward_func(return_:Tensor, action:Tensor, ema_return:float, ema_sq_return:float, eta: float = 0.01) -> tuple:
    action = torch.where(action == 2, -1, action.int())
    fee_penalty = 0.001 if action != 0 else 0
    net_return = return_.item() - fee_penalty

    A_prev = ema_return
    B_prev = ema_sq_return

    variance:float = B_prev - (A_prev ** 2)
    if variance < 1e-8:
        variance = 1e-8

    delta_A = net_return - A_prev
    delta_B = (net_return ** 2) - B_prev

    numerator = (B_prev * delta_A) - (0.5 * A_prev * delta_B)
    denominator = math.pow(variance, 1.5)
    dsr = numerator / denominator

    ema_return = A_prev + eta * delta_A
    ema_sq_return = B_prev + eta * delta_B

    reward = np.clip(dsr, -1.0, 1.0)
    return (reward.item(), ema_return, ema_sq_return)

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

def max_dd(portfolio: list[float], dd:bool=False) -> Tensor:
    values = torch.tensor(portfolio)
    if values.shape[-1] < 2: return torch.tensor(0.0)
    if dd:
        peak = torch.max(values)
        drawdowns = (peak - values[-1]) / (peak + 1e-9)  
        return drawdowns
    peak = torch.cummax(values, dim=0).values
    drawdowns = (peak - values) / (peak + 1e-9)
    mdd = torch.max(drawdowns)
    return mdd

def calcul_cost(amount: Tensor, cost_rate: Tensor, device:str) -> Tensor: return amount * cost_rate

#def compute_entropy(prob: list[float]): return entropy(prob, base=2)

def profit_and_loss(total_price:Tensor) -> Tensor: return  total_price[2] - total_price[3]

def return_log_vec(data:list[float], device:str) -> Tensor:
    data_t = torch.as_tensor(data, dtype=torch.float32, device=device)
    p_return = torch.log(data_t[1:]/data_t[:-1])
    return p_return

#Compute the return log
def return_log(data: Tensor, device:str) -> Tensor: 
    if data[1] > 1e-8 and data[0] > 1e-8: 
        return torch.log(data[1] / data[0])
    return torch.tensor(0.0, device=device)
