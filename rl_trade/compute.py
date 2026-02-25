import pandas as pd
import numpy as np
from scipy.stats import entropy
import math
import torch
from torch import Tensor
from collections import deque

BETA = .2

def reward_func(return_:Tensor, action:Tensor, p_value_hist: deque[float]) -> float:
    action = torch.where(action == 2, -1, action.int())
    trade_executed = False
    if action != 0: trade_executed = True
    if torch.isnan(return_) or torch.isinf(return_): return -1.0 
    
    step_return = return_.item()
    
    # Paramètres fixes et robustes
    alpha = 0.88
    beta = 0.88
    # On met une aversion globale très forte (ex: 3.0 ou 4.0 au lieu de 2.25)
    lmbda = 3.5 
    
    # Calcul strict de l'utilité
    if step_return >= 0:
        v_x = step_return ** alpha
    else:
        v_x = -lmbda * (abs(step_return) ** beta)

    fee_penalty = 0.001 if trade_executed else 0.0
    
    # Drawdown Penalty exponentiel
    current_dd = max_dd(p_value_hist, True).item()
    dd_penalty = 0.0
    if current_dd > 0.05: # Tolérance très basse (5%)
        dd_penalty = (current_dd * 2.0) ** 2 # La douleur monte au carré 

    reward = v_x - fee_penalty - dd_penalty
    return float(np.clip(reward, -1.0, 1.0))

#Compute the sharpe ration
def calcul_sharpe_ratio(portfolio_value: list) -> float:
    if len(portfolio_value) < 2: return 0.0
    val_arr = np.array(portfolio_value, dtype=np.float64)
    
    returns = np.diff(val_arr) / (val_arr[:-1] + 1e-9)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    std_dev = np.std(returns)
    if std_dev > 1e-8:
        sharpe = (np.mean(returns) / std_dev) * np.sqrt(365 * 24)
        if np.isnan(sharpe) or np.isinf(sharpe):
            return 0.0
        return float(sharpe)
    else:
        return 0.0

def max_dd(portfolio: deque[float], dd:bool=False) -> Tensor:
    values = torch.tensor(list(portfolio), dtype=torch.float32)
    if values.shape[0] < 2: return torch.tensor(0.0)
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
