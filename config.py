import torch
import pandas as pd
from dataclasses import dataclass, field


# PPO hyper-params
@dataclass(frozen=True)
class PPOConfig:
    lr: float = 3e-5
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_eps: float = 0.1
    ent_coef: float = 0.01
    value_coef: float = 0.5
    belief_coef: float = 0.1
    change_coef: float = 0.1


#Macro_head train config
@dataclass
class MacroConfig:
    lr: float = 0.003850091902319146
    epochs: int = 30
    class_weights_belief: tuple[float, float, float] = field(default_factory=lambda: (1., 0.8835228721407905, 0.9362682099573649))
    class_weights_change: tuple[float, float] = field(default_factory=lambda: (1., 0.981713583978431))


#Training default config
@dataclass
class TrainConfig:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    init_amount: float = 100_000.0
    data_path: str = "./data_off/train_test"
    data_train: dict[str, pd.DataFrame] = field(default_factory=dict)
    data_pretrain: dict[str, pd.DataFrame] = field(default_factory=dict)
    model_path: str = "./agent/save"
    timestamp: int = 6_000_000
    batch_size: int = 64
    rollout_steps: int = 2048
    num_update: int = field(init=False)

    def __post_init__(self) -> None:
        self.data_train ={
                    "micro": pd.read_csv(f"{self.data_path}/daily_train.csv").iloc[:, 1:],
                    "macro": pd.read_csv(f"{self.data_path}/metric_train.csv").iloc[:, 1:],
                    "price": pd.read_csv(f"{self.data_path}/price_close_train.csv")["Close"],
                    "regime": pd.read_csv(f"{self.data_path}/regime_train.csv")["regime"],
                    "change": pd.read_csv(f"{self.data_path}/change_train.csv")["change"]
                    }
        self.data_pretrain = {
                    "feature": pd.read_csv(f"{self.data_path}/metric_pretrain.csv").iloc[:, 1:],
                    "regime": pd.read_csv(f"{self.data_path}/regime_pretrain.csv").iloc[:, 1:],
                    "change": pd.read_csv(f"{self.data_path}/change_pretrain.csv").iloc[:, 1:],
                    }
        self.num_update = self.timestamp // self.rollout_steps
        print(f"{self.device} is used")


#Wandb default config
@dataclass
class WandbConfig:
    name: str = "Kairos"
    logs: dict[str, float] = field(default_factory=dict)
