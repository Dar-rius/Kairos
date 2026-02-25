import gymnasium as gym
from typing import Any
import torch
import numpy as np
from .env import *

#This class convert all data stored in tensor to numpy
class TensorToNumpyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def _convert_to_numpy(self, obs) -> np.array:
        if isinstance(obs, torch.Tensor):
            return obs.detach().cpu().numpy()
        return obs

    def reset(self, **kwargs) -> np.array:
        #  call reset function
        obs = self.env.reset(**kwargs)
        return self._convert_to_numpy(obs)

    def step(self, action) -> Any:
        if isinstance(action, (np.ndarray, int, float)):
            action_tensor = torch.tensor(action, device=self.env.device)
        else:
            action_tensor = action

        obs, reward, _, truncated, done = self.env.step(action_tensor)

        # Convert obs, truncated and done 
        obs = self._convert_to_numpy(obs)

        return obs, reward, done.item(), truncated.item()
