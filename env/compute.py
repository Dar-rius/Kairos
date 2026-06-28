import math
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from collections import deque

def reward_func(return_:float, belief_probs:float, position:int, prev_position:int, step:int, dsr_nu:float, ema_a:float, ema_b:float, alpha:float, gamma:float, beta:float) -> tuple[float, float, float]:
    # Calcul delta A and B
    delta_a =  return_ - ema_a
    delta_b =  (return_**2) - ema_b
    # Calcul A and B
    new_ema_a = ema_a + (dsr_nu * delta_a)
    new_ema_b = ema_b + (dsr_nu * delta_b)
    # if step is less than 4
    if step < 10: return np.clip(return_*50, -5.0, 5.0).item(), new_ema_a, new_ema_b
    epsilon = 1e-4
    variance = ema_b - (ema_a ** 2)
    
    # Calcul DSR
    numerator = (ema_b * delta_a) - (0.5 * ema_a * delta_b)
    denominator = np.pow(np.clip(variance, min=0.0) + epsilon, 1.5)
    dsr = numerator / denominator
    
    #belief_entropy = -torch.sum(belief_probs * torch.log(belief_probs + 1e-8), dim=-1)
    #max_entropy = torch.log(torch.tensor(3.0))
    #confidence = alpha * (1.0 - (belief_entropy / max_entropy))
    #stability = beta * (dsr * 10.0)
    #return_win = confidence * (position * (return_*10.0))
    #penality = gamma * abs(position - prev_position)
    #total_reward = stability + return_win - penality

    reward = np.clip(dsr*10.0, -5.0, 5.0).item()
    #print(f"Step: {step} | Return brut: {return_.item()} | DSR: {dsr} | Confidence: {confidence}| Total brute: {reward} ")
    return reward, new_ema_a, new_ema_b


def calcul_sharpe_ratio(p_value:np.ndarray) -> float:
    if p_value.shape[0] < 2: return 0.0
    if not np.isfinite(p_value).all(): return 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = np.diff(p_value) / (p_value[:-1] + 1e-9)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    std_dev = np.std(returns)
    if std_dev > 1e-8:
        sharpe = (np.mean(returns) / std_dev) * np.sqrt(365 * 24)
        if np.isnan(sharpe) or np.isinf(sharpe): return 0.0
        return float(sharpe)
    return 0.0


def calcul_max_dd(p_value:np.ndarray, dd:bool=False) -> float:
    if p_value.shape[0] < 2: return 0.0
    if not np.isfinite(p_value).all(): return 1.0
    if dd:
        peak = np.max(p_value)
        drawdowns = (peak - p_value[-1]) / (peak + 1e-9)
        return drawdowns.item()
    peak = np.maximum.accumulate(p_value, axis=0)
    drawdowns = (peak - p_value) / (peak + 1e-9)
    mdd = np.max(drawdowns)
    return mdd.item()

# Calcul the transfer's cost
def calcul_cost(amount: float, cost_rate: float) -> float: return amount * cost_rate

# Calcul the benefice final of trade
def profit_and_loss(all_price:list) -> list: return  all_price[2] - all_price[3]

# Cacul return
def return_log(data: list[float]) -> float:
    p_return = math.log(data[1]/data[0])
    return p_return

def convert_to_btc(amount_usd: float, btc_value: float) -> float: return amount_usd / btc_value

def convert_to_usd(amount_btc: float, btc_value: float) -> float: return amount_btc * btc_value
