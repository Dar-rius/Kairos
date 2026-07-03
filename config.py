import torch
import pandas as pd
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

@dataclass
class TrainingConfig:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    data_path: str = "./data_off/train_test"
    data_train: dict[str, pd.DataFrame] = {
            "micro": pd.read_csv(f"{data_path}/daily_train.csv").iloc[:, 1:],
            "macro": pd.read_csv(f"{data_path}/metric_train.csv").iloc[:, 1:],
            "price": pd.read_csv(f"{data_path}/price_close_train.csv")["Close"],
            "regime": pd.read_csv(f"{data_path}/regime_train.csv")["regime"],
            "change": pd.read_csv(f"{data_path}/change_train.csv")["change"]
            }
    data_pretrain: dict[str, pd.DataFrame] = {
            "feature": pd.read_csv(f"{data_path}/metirc_pretrain.csv").iloc[:, 1:],
            "belief": pd.read_csv(f"{data_path}/regime_pretrain.csv").iloc[:, 1:],
            "regime": pd.read_csv(f"{data_path}/change_pretrain.csv").iloc[:, 1:],
            }
    model_path: str = "./agent/save"
    total_timestamp: int = 6_000_000
    batch_size: int = 64
    rollout_steps: int = 2048
    num_update: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_update = self.total_timestamp // self.rollout_steps

@dataclass(frozen=True)
class MacroConfig:
    lr: float = 0.003850091902319146
    epochs: int = 30
    class_weights_belief: list = [1., 0.8835228721407905, 0.9362682099573649] 
    class_weights_change: list = [1., 0.981713583978431]


@dataclass
class WandbConfig:
    name: str = "Kairos"
    logs: dict = field(default_factory=dict)
