import torch
import numpy as np

class Buffer:
    """
    0 -> Micro State 
    1 -> Macro State
    2 -> Action
    3 -> Old Log Probs 
    4 -> Return
    5 -> Advantage
    6 -> Target Regime
    """
    def __init__(self, size: int, device:str):
        self.size = size
        self.micro_states = torch.zeros((self.size, 24, 6), device, dtype = torch.float32)
        self.macro_states = torch.zeros((self.size, 7), device, dtype = torch.float32)
        self.actions = torch.zeros((self.size, 1), device, dtype = torch.int32)
        self.old_log_probs = torch.zeros((self.size, 1), device, dtype = torch.float32)
        self.returns = torch.zeros((self.size, 1), device, dtype = torch.float32)
        self.target_regimes = torch.zeros((self.size, 1), device, dtype = torch.int32)
        self.slice: int = 0

    def insert_data(self, micro_state: np.array, macro_state: np.array, actions: int, old_log_probs: list,  returns: float, target_regimes:int):
        self.micro_states[self.slice] = torch.from_numpy(micro_state)
        self.macro_states[self.slice] = torch.from_numpy(macro_state)
        self.actions[self.slice] = actions
        self.old_log_probs[self.slice] = old_log_probs
        self.returns[self.slice] = returns
        self.target_regimes[self.slice] = target_regimes
        self.slice += 1

    # sampling data
    def sample(self) -> tuple:
        return (self.micro_states, self.macro_states,
                self.actions, self.old_log_probs,
                self.returns, self.target_regimes)

    # Delete all data
    def clear(self):
        self.micro_states.zero_()
        self.macro_states.zero_()
        self.actions.zero_()
        self.old_log_probs.zero_()
        self.returns.zero_()
        self.target_regimes.zero_()
