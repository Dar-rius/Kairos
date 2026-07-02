import torch
from dataclasses import dataclass, field

@dataclass(frozen=True)
class PPOConfig:
    lr: float = 3e-5
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_eps: float = 0.1
    ent_coef: float = 0.001
    value_coef: float = 0.5
    belief_coef: float = 0.1
    change_coef: float = 0.1

@dataclass(frozen=True)
class TrainConfig:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    data_path: str = "./data_off/train_test/"
    model_path: str = "./agent/save"
    total_timestamp: int = 6_000_000
    batch_size: int = 64
    rollout_steps: int = 2048
    num_update: int = total_timestamp // rollout_steps

@dataclass(frozen=True)
class WandbConfig:
    name: str = "Kairos"
    logs: dict = field(default_factory=dict)
