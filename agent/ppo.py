import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import Tensor
from .buffer import Buffer
from .model import Agent
from torch.utils.tensorboard import SummaryWriter

# Class for TensorBoard
class Writer:
    def __init__(self, path:str):
        self.writer = SummaryWriter(log_dir=path)

    def add(self, step, policy_loss:float, critic_loss:float, entropy_loss:float, belief_loss:float, loss:float, reward:float, pnl:float, sharpe:float, mdd:float, hold_pct:float, buy_pct:float, sell_pct:float):
        self.writer.add_scalar("Policy Loss", policy_loss, step)
        self.writer.add_scalar("Critic Loss", critic_loss, step)
        self.writer.add_scalar("Belief Loss", belief_loss, step)
        self.writer.add_scalar("Entropy Loss", entropy_loss, step)
        self.writer.add_scalar("Loss", loss, step)
        self.writer.add_scalar("Reward", reward, step)
        self.writer.add_scalar("PNL", pnl, step)
        self.writer.add_scalar("Sharpe ratio", sharpe, step)
        self.writer.add_scalar("Max Draw Down", mdd, step)
        self.writer.add_scalar("Hold", hold_pct, step)
        self.writer.add_scalar("Buy", buy_pct, step)
        self.writer.add_scalar("Sell", sell_pct, step)
    def close(self): self.writer.close()

# Belief PPO Implementation
class PPOTrainer:
    def __init__(self, model:Agent, device:str, lr:float=3e-4, gamma:float=0.99, gae_lambda:float=0.95, clip_eps:float=0.2, value_coef:float=0.5, belief_coef:float=0.5, ent_coef:float=0.01):
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
        self.ent_coef = ent_coef
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.device = device

    def compute_gae(self, rewards:Tensor, values:Tensor, last_value:Tensor, dones:Tensor) -> tuple[Tensor, Tensor]:
        gae: float = 0.0
        mask = 1.0 - dones
        next_values = torch.cat((values[1:], last_value), 0)
        total_size = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        delta = rewards + self.gamma * next_values * mask - values
        for step in reversed(range(total_size)):
            gae = delta[step] + self.gamma * self.gae_lambda * mask[step] * gae
            advantages[step] =  gae
        returns = advantages + values
        return (returns, advantages)

    def lr_decay(self, lr:float, total_steps:int, step:int):
        frac = 1.0 - (step / total_steps)
        current_lr = lr * frac
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

    # Compute Belief PPO and Update network weights
    def update(self, memory:Buffer, total_steps:int, step:int, batch_size:int=64, epochs:int=10):
        self.lr_decay(self.lr, total_steps, step)
        # the target regime (0 -> Stable, 1 -> Volatility, 2 -> Crisis)
        micro_states, macro_states, actions, old_log_probs, returns, adv, _, _, _, target_regimes = memory.get_all()
        # Normalize the advantages
        advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
        dataset_size = actions.size(0)
        for _ in range(epochs):
            all_indices = torch.randperm(dataset_size, device=self.device)
            for start in (0, dataset_size, batch_size):
                end = start + batch_size
                idx = all_indices[start:end]
                if idx.numel() == 0: continue
                # Evaluate model again
                _, new_log_probs, dist_entropy, new_values, belief_logits, belief_entropy = self.model.get_action_and_value(micro_states[idx], macro_states[idx], actions[idx])
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
                entropy_loss = dist_entropy.mean()
                # Total Loss
                loss = policy_loss + \
                       (self.value_coef * value_loss) + \
                       (self.belief_coef * belief_loss) - \
                       (self.ent_coef * entropy_loss)
                # Backpropagation
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
        return loss.item(), policy_loss.item(), value_loss.item(), belief_loss.item(), dist_entropy.mean().item()
