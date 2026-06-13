import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import Tensor
from .buffer import Buffer
from .model import Agent, FocalLoss

# Belief PPO Implementation
class PPOTrainer:
    def __init__(self,
                 model:Agent,
                 lr:float=3e-4,
                 gamma:float=0.99,
                 gae_lambda:float=0.95,
                 clip_eps:float=0.2,
                 value_coef:float=0.5,
                 belief_coef:float=0.1,
                 change_coef:float=0.1,
                 ent_coef:float=0.01, 
                 device:str="cpu"
                 ):
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
        self.ent_coef_end = 0.1
        self.mse_loss = nn.MSELoss()
        self.fl_loss = FocalLoss()
        self.bfl_loss = FocalLoss()
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

    def compute_complexity(self, action_probs: torch.Tensor) -> torch.Tensor:
        # Entropie H
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
        # Disequilibrium D = sum (p - 1/|A|)^2
        target_uniform = 1.0 / 3.0
        disequilibrium = torch.sum((action_probs - target_uniform) ** 2, dim=-1)
        # Complexitt C = H * D
        complexity = entropy * disequilibrium
        return complexity.mean()

    #def gradient_diagnostic(self, loss_value:Tensor, loss_policy:Tensor, loss_belief:Tensor):
        

    # Compute Belief PPO and Update network weights
    def update(self, memory:Buffer, total_steps:int, step:int, batch_size:int=64, epochs:int=10):
        self.lr_decay(self.lr, total_steps, step)
        # the target regime (0 -> Stable, 1 -> Volatility, 2 -> Crisis)
        micro_states, macro_states, pos_type, actions, old_log_probs, returns, adv, _, _, _, target_regimes, target_changes, _, _ = memory.get_all()
        # Normalize the advantages
        advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
        dataset_size = actions.size(0)
        num_batch = dataset_size // batch_size
        size_total = int((dataset_size / batch_size) * epochs)
        epoch_losses = torch.zeros((size_total), dtype=torch.float32, device=self.device)
        epoch_pi_losses = torch.zeros((size_total), device=self.device)
        epoch_v_losses = torch.zeros((size_total), device=self.device)
        epoch_b_losses = torch.zeros((size_total), device=self.device)
        epoch_c_losses = torch.zeros((size_total), device=self.device)
        epoch_entropies = torch.zeros((size_total), device=self.device)
        epoch_complexity = torch.zeros((size_total), device=self.device)
        index_loss = 0
        batch_rollout = torch.arange(0, dataset_size, batch_size, device=self.device)
        for _ in range(epochs):
            shuffle_index = batch_rollout[torch.randperm(num_batch, device=self.device)]
            for start in shuffle_index:
                end = start + batch_size
                idx = torch.arange(start, end, device=self.device)
                if idx.numel() == 0: continue
                # Evaluate model again
                _, new_log_probs, dist_entropy, new_values, belief_logits, change_logits, _,  _, actor_logits = self.model.get_action_and_value(micro_states[idx], macro_states[idx], pos_type[idx], actions[idx])
                # Compute Ratio (new Policy / old Policy)
                logratio = new_log_probs - old_log_probs[idx]
                ratio = torch.exp(logratio)
                #compute other
                action_probs = torch.softmax(actor_logits, dim=-1)
                #update the complexity
                complexity = self.compute_complexity(action_probs)
                # Loss PPO
                idx_adv = advantages[idx].flatten()
                surr1 = ratio * idx_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * idx_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                # Loss Value (Critic) - MSE
                value_loss = self.mse_loss(new_values.flatten(), returns[idx].flatten())
                # Loss Belief (Auxiliary) - Cross Entropy
                belief_loss = self.fl_loss(belief_logits, target_regimes[idx].flatten().long())
                change_loss = self.bfl_loss(change_logits, target_changes[idx].flatten().long())
                entropy_loss = dist_entropy.mean()
                # Total Loss
                loss = policy_loss + \
                        (self.value_coef * value_loss) + \
                        (self.belief_coef * belief_loss) + \
                        (self.change_coef * change_loss) - \
                        (self.ent_coef_end * complexity)
                # Backpropagation
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_losses[index_loss] = loss
                epoch_pi_losses[index_loss] = policy_loss
                epoch_v_losses[index_loss] = value_loss
                epoch_b_losses[index_loss] = belief_loss
                epoch_c_losses[index_loss] = change_loss
                epoch_entropies[index_loss] = entropy_loss
                epoch_complexity[index_loss] = complexity
        return epoch_losses.mean().item(), epoch_pi_losses.mean().item(), epoch_v_losses.mean().item(), epoch_b_losses.mean().item(), epoch_c_losses.mean().item(), epoch_entropies.mean().item(), epoch_complexity.mean().item()
