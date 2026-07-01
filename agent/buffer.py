import torch
import numpy as np
from torch import Tensor

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
    10 -> Target Change
    """
    def __init__(self, step:int, micro_size:int, macro_size:int, device:str):
        self.step = step
        self.slice: int = 0
        self.device = device
        self.micro_states = np.zeros((self.step, 24, micro_size))
        self.macro_states = np.zeros((self.step, macro_size))
        self.pos_type = np.zeros((self.step, 3))
        self.actions = np.zeros((self.step))
        self.old_log_probs = np.zeros(self.step)
        self.returns = np.zeros(self.step)
        self.adv = np.zeros(self.step)
        self.target_regimes = np.zeros(self.step)
        self.target_changes = np.zeros(self.step)
        self.rewards = np.zeros(self.step)
        self.values = np.zeros(self.step)
        self.dones = np.zeros(self.step)
        self.beliefs = np.zeros((self.step, 3))
        self.portfolio = np.zeros((self.step,1))

    def insert(self, 
               micro_state:np.ndarray,
               macro_state:np.ndarray,
               pos_type:list,
               action:int,
               old_log_prob:float,
               reward:float,
               value:float,
               dones:int,
               target_regime:int,
               target_change:int,
               beliefs:int,
               portfolio: float):
        self.micro_states[self.slice] = micro_state
        self.macro_states[self.slice] = macro_state
        self.pos_type[self.slice] = pos_type
        self.actions[self.slice] = action
        self.old_log_probs[self.slice] = old_log_prob
        self.rewards[self.slice] = reward
        self.values[self.slice] = value
        self.dones[self.slice] = dones
        self.target_regimes[self.slice] = target_regime
        self.target_changes[self.slice] = target_change
        self.beliefs[self.slice] = beliefs
        self.portfolio[self.slice] = portfolio
        self.slice += 1

    def insert_returns(self, returns:np.ndarray, adv:np.ndarray):
        self.returns[:] = returns
        self.adv[:] = adv
    
    # sampling data
    def get_all(self) -> tuple:
        return (torch.tensor(self.micro_states, dtype=torch.float32, device=self.device),
                torch.tensor(self.macro_states, dtype=torch.float32, device=self.device),
                torch.tensor(self.pos_type, dtype=torch.long, device=self.device),
                torch.tensor(self.actions, dtype=torch.long, device=self.device),
                torch.tensor(self.old_log_probs, dtype=torch.float32, device=self.device),
                torch.tensor(self.returns, dtype=torch.float32, device=self.device),
                torch.tensor(self.adv, dtype=torch.float32, device=self.device),
                torch.tensor(self.rewards, dtype=torch.float32, device=self.device),
                torch.tensor(self.values, dtype=torch.float32, device=self.device),
                torch.tensor(self.dones, dtype=torch.long, device=self.device),
                torch.tensor(self.target_regimes, dtype=torch.long, device=self.device),
                torch.tensor(self.target_changes, dtype=torch.long,  device=self.device),
                torch.tensor(self.beliefs, dtype=torch.long, device=self.device),
                torch.tensor(self.portfolio, dtype=torch.float32, device=self.device))

    # Delete all data
    def clear(self):
        self.slice = 0
