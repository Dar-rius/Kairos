import torch
from torch import Tensor
import numpy as np

class Buffer:
    """
    0 -> Micro State 
    1 -> Macro State
    2 -> Action
    3 -> Old Log Probs 
    4 -> Returns
    5 -> Advantage
    6 -> Reward
    7 -> Value
    8 -> Dones
    9 -> Target Regime
    """
    def __init__(self, step:int, micro_size:int, macro_size:int, device:str):
        self.step = step
        self.slice: int = 0
        self.device = device
        self.micro_states = torch.zeros((self.step, 23, micro_size), device=self.device)
        self.macro_states = torch.zeros((self.step, macro_size), device=self.device)
        self.actions = torch.zeros((self.step), dtype=torch.long, device=self.device)
        self.old_log_probs = torch.zeros((self.step), device=self.device)
        self.returns = torch.zeros((self.step), device=self.device)
        self.adv = torch.zeros((self.step), device=self.device)
        self.target_regimes = torch.zeros((self.step), dtype=torch.long, device=self.device)
        self.rewards = torch.zeros((self.step), device=self.device)
        self.values = torch.zeros((self.step), device=self.device)
        self.dones = torch.zeros((self.step), device=self.device)

    def insert(self, micro_state: Tensor, macro_state: Tensor, action: int, old_log_prob: list,  reward: float, value:float, dones:float, target_regime:int):
        self.micro_states[self.slice] = micro_state
        self.macro_states[self.slice] = macro_state
        self.actions[self.slice] = action
        self.old_log_probs[self.slice] = old_log_prob
        self.rewards[self.slice] = reward
        self.values[self.slice] = value
        self.dones[self.slice] = dones
        self.target_regimes[self.slice] = target_regime
        self.slice += 1

    def insert_returns(self, returns:Tensor, adv:Tensor):
        self.returns[:] = returns
        self.adv[:] = adv
    
    # sampling data
    def get_all(self) -> tuple:
        return (self.micro_states, self.macro_states,
                self.actions, self.old_log_probs,
                self.returns, self.adv, self.rewards, 
                self.values, self.dones, self.target_regimes)

    # Delete all data
    def clear(self):
        self.slice = 0
