import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import Tensor
from .buffer import Buffer
from .model import Agent

# Belief PPO Implementation
class PPOTrainer:
    def __init__(self, model:Agent, device:str, lr:float=3e-4, gamma:float=0.99, gae_lambda:float=0.95, clip_eps:float=0.2, value_coef:float=0.5, belief_coef:float=0.5, change_coef:float=0.5, ent_coef:float=0.01):
        self.model = model
        self.lr = lr
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Hyperparams PPO
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        # Total Loss Coefficients
        self.value_coef = value_coef
        self.belief_coef = belief_coef
        self.change_coef = change_coef
        self.ent_coef = ent_coef
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.CrossEntropyLoss()
        self.device = device
        self.A_hat = 0.0
        self.t = 0

    def compute_gae(self, rewards:Tensor, values:Tensor, last_value:Tensor, dones:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        gae: Tensor = torch.tensor(0.0)
        mask = 1.0 - dones
        next_values = torch.cat((values[1:], last_value), 0)
        total_size = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        delta = rewards + self.gamma * next_values * mask - values
        for step in reversed(range(total_size)):
            gae = delta[step] + self.gamma * self.gae_lambda * mask[step] * gae
            advantages[step] =  gae
        returns = advantages + values
        return (returns, advantages, delta)

    def lr_decay(self, lr:float, total_steps:int, step:int):
        frac = 1.0 - (step / total_steps)
        current_lr = lr * frac
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

    def update_aes(self, td_errors:Tensor) -> float:
        self.t += 1
        abs_errors = torch.abs(td_errors)
        alpha = torch.quantile(abs_errors, .9)
        self.A_hat += alpha.item()
        lambda_t_raw = np.sqrt(self.A_hat / self.t)
        lambda_t = float(np.clip(lambda_t_raw, 0.01, 0.1))
        return lambda_t

    def reset_aes(self):
        self.A_hat = 0.0
        self.t = 0

    # Compute Belief PPO and Update network weights
    def update(self, memory:Buffer, total_steps:int, step:int, batch_size:int=64, epochs:int=10):
        self.lr_decay(self.lr, total_steps, step)
        # the target regime (0 -> Stable, 1 -> Volatility, 2 -> Crisis)
        micro_states, macro_states, pos_type, actions, old_log_probs, returns, adv, _, _, _, target_regimes, target_changes, _, _ = memory.get_all()
        # Normalize the advantages
        advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
        dataset_size = actions.size(0)
        all_indices = torch.randperm(dataset_size, device=self.device)
        for _ in range(epochs):
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = all_indices[start:end]
                if idx.numel() == 0: continue
                # Evaluate model again
                _, new_log_probs, dist_entropy, new_values, belief_logits, change_logits, _,  _= self.model.get_action_and_value(micro_states[idx], macro_states[idx], pos_type[idx], actions[idx])
                with torch.no_grad():
                    vals = new_values.flatten()
                    delta_proxy = returns[idx].flatten() - vals
                lambda_t = self.update_aes(delta_proxy)
                # Compute Ratio (new Policy / old Policy)
                logratio = new_log_probs - old_log_probs[idx]
                ratio = torch.exp(logratio)
                # Loss PPO
                idx_adv = advantages[idx].flatten()
                surr1 = ratio * idx_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * idx_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                # Loss Value (Critic) - MSE
                value_loss = self.mse_loss(new_values.flatten(), returns[idx].flatten())
                # Loss Belief (Auxiliary) - Cross Entropy
                belief_loss = self.ce_loss(belief_logits, target_regimes[idx].flatten().long())
                change_loss = self.bce_loss(change_logits, target_changes[idx].flatten().long())
                entropy_loss = dist_entropy.mean()
                # Total Loss
                loss = policy_loss + \
                       (self.value_coef * value_loss) + \
                       (self.belief_coef * belief_loss) + \
                       (self.change_coef * change_loss) - \
                       (0.02 * entropy_loss)
                # Backpropagation
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
        return loss.item(), policy_loss.item(), value_loss.item(), belief_loss.item(), change_loss.item(), dist_entropy.mean().item()
