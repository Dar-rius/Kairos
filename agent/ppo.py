import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .buffer import Buffer
from .model import Agent
from torch.utils.tensorboard import SummaryWriter

# Class for TensorBoard
class Writer:
    def __init__(self, path:str):
        self.writer = SummaryWriter(log_dir=path)

    def add(self, step, policy_loss:float, critic_loss:float, entropy_loss:float, belief_loss:float, loss:float, reward:float, pnl:float):
        self.writer.add_scalar("Policy Loss", policy_loss, step)
        self.writer.add_scalar("Critic Loss", critic_loss, step)
        self.writer.add_scalar("Belief Loss", belief_loss, step)
        self.writer.add_scalar("Entropy Loss", entropy_loss, step)
        self.writer.add_scalar("Loss", loss, step)
        self.writer.add_scalar("Reward", reward, step)
        self.writer.add_scalar("PNL", pnl, step)

    def close(self): self.writer.close()

# Belief PPO Implementation
class PPOTrainer:
    def __init__(self, model:Agent, lr:float=3e-4, gamma:float=0.99, gae_lambda:float=0.95, clip_eps:float=0.2, value_coef:float=0.5, belief_coef:float=0.5, ent_coef:float=0.01):
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

    def compute_gae(self, rewards:list, values:list, last_value:float, dones:list) -> np.array:
        values = values + [last_value]
        returns: list(float) = []
        gae: float = 0.0
        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + self.gamma * values[step + 1] * mask - values[step]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            returns.insert(0,  gae + values[step])
        returns = np.array(returns).reshape(-1,1)
        return returns

    def lr_decay(self, lr:float, total_steps:int, step:int):
        frac = 1.0 - (step / total_steps)
        current_lr = lr * frac
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

    # Compute Belief PPO and Update network weights
    def update(self, memory:Buffer, total_steps:int, step:int, batch_size:int=64, epochs:int=10):
        self.lr_decay(self.lr, total_steps, step)
        # the target regime (0 -> Stable, 1 -> Volatility, 2 -> Crisis)
        micro_states, macro_states, actions, old_log_probs, returns, _, _, _, target_regimes = memory.get_all()
        # Normalize the advantages
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        dataset_size = len(actions)
        indices = np.arange(0, dataset_size, batch_size)
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in indices:
                end = start + batch_size
                idx = slice(start,end)
                # Evaluate model again
                _, new_log_probs, dist_entropy, new_values, belief_logits, belief_entropy = self.model.get_action_and_value(micro_states[idx], macro_states[idx], actions[idx])
                # Compute Ratio (new Policy / old Policy)
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                # Loss PPO
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                # Loss Value (Critic) - MSE
                value_loss = self.mse_loss(new_values, returns[idx])
                # Loss Belief (Auxiliary) - Cross Entropy
                belief_loss = self.ce_loss(belief_logits, target_regimes[idx].view(-1).long())
                # Total Loss
                loss = policy_loss + \
                       (self.value_coef * value_loss) + \
                       (self.belief_coef * belief_loss) - \
                       (self.ent_coef * dist_entropy.mean())
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
        return loss.item(), policy_loss.item(), value_loss.item(), belief_loss.item(), dist_entropy.mean().item()
