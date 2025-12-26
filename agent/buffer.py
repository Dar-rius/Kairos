import torch
import numpy as np

class Buffer:
    """
    0 -> Micro State 
    1 -> Macro State
    2 -> Action
    3 -> Old Log Probs 
    4 -> Reward
    5 -> Value
    6 -> Target Regime

    Other: Return (Advantage)
    """
    def __init__(self, step:int, micro_size:int, macro_size:int):
        self.step = step
        self.slice: int = 0
        self.micro_states = np.zeros((self.step, 24, micro_size), dtype = torch.float32)
        self.macro_states = np.zeros((self.step, macro_size), dtype = torch.float32)
        self.actions = np.zeros((self.step, 1), dtype = torch.int32)
        self.old_log_probs = np.zeros((self.step, 1), dtype = torch.float32)
        self.returns = np.zeros((self.step, 1), dtype = torch.float32)
        self.rewards = np.zeros((self.step, 1), dtype = torch.float32)
        self.values = np.zeros((self.step, 1), dtype=torch.float32)
        self.dones = np.zeros((self.step, 1), dtype=torch.float32)
        self.target_regimes = np.zeros((self.step, 1), dtype = torch.int32)

    def insert(self, micro_state: np.array, macro_state: np.array, action: int, old_log_prob: list,  reward: float, value:float, dones:float, target_regime:int):
        self.micro_states[self.slice] = micro_state
        self.macro_states[self.slice] = macro_state
        self.actions[self.slice] = action
        self.old_log_probs[self.slice] = old_log_prob
        self.rewards[self.slice] = reward
        self.values[self.slice] = value
        self.dones[self.slice] = dones
        self.target_regimes[self.slice] = target_regime
        self.slice += 1

    def insert_returns(self, returns: list(float)):
        self.returns[:] = returns 
    
    # sampling data
    def get_all(self) -> tuple:
        return (self.micro_states, self.macro_states,
                self.actions, self.old_log_probs,
                self.returns, self.rewards, 
                self.values, self.dones, self.target_regimes)

    # Delete all data
    def clear(self):
        self.slice = 0
