import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import Tensor
import numpy as np
class MacroHead(nn.Module):
    def __init__(self, macro_dim:int, num_regimes:int=3, num_changes:int=2):
        super(MacroHead, self).__init__()
        self.macro_net = nn.Sequential(
            nn.Linear(macro_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.belief_head = nn.Linear(32, num_regimes)
        self.change_head = nn.Linear(32, num_changes)
        self._init_weights()

    def _init_weights(self):
        for layer in self.macro_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

        nn.init.orthogonal_(self.belief_head.weight, gain=1.0)
        nn.init.constant_(self.belief_head.bias, 0.0)
        
        nn.init.orthogonal_(self.change_head.weight, gain=1.0)
        nn.init.constant_(self.change_head.bias, 0.0)

    def forward(self, macro_x:np.ndarray):
        x = self.macro_net(macro_x)
        belief_logits = self.belief_head(x)
        change_logits = self.change_head(x)
        return x, belief_logits, change_logits

class Agent(nn.Module):
    def __init__(self, pretrained_model:MacroHead, micro_dim:int, action_dim:int, num_regimes:int=3, num_change:int=2):
        super(Agent, self).__init__()
        self.belief_head = pretrained_model
        self.micro_lstm = nn.LSTM(micro_dim, 128, batch_first=True)
        
        # --- FUSION HIÉRARCHIQUE ---
        # 128 (Micro) +  3 (Macro Explicit Prediction) +  2 (change state) + 3 (state actions) + 1 (portfolio value)
        fusion_dim = 128 + num_regimes + num_change + 3 + 1

        self.actor_layer = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim))
        
        self.critic = nn.Linear(fusion_dim, 1)
        
        self._init_weights()

    def _init_weights(self):
        # Initialize LSTM
        for name, param in self.micro_lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

        for layer in self.actor_layer:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

        actor_out = self.actor_layer[-1]
        nn.init.orthogonal_(actor_out.weight, gain=0.01)
        nn.init.constant_(actor_out.bias, 0.0)

    def forward(self, micro_x:Tensor, macro_x:Tensor, pos_type:Tensor, p_value:Tensor):
        # System 2
        _ , belief_logits, change_logits = self.belief_head(macro_x)
        current_belief_probs = torch.softmax(belief_logits, dim=1)
        current_change_probs = torch.sigmoid(change_logits)

        # SYSTEM 1
        self.micro_lstm.flatten_parameters()
        _, (h_n, _) = self.micro_lstm(micro_x)
        micro_feat = h_n[-1]
        # FUSION (context)
        if p_value.dim() == 1:
            p_value = p_value.unsqueeze(1)
        context = torch.cat([micro_feat, current_belief_probs, current_change_probs, pos_type, p_value], dim=1)
        action_logits = self.actor_layer(context)
        value = self.critic(context)
        return action_logits, value, belief_logits, change_logits

    def get_action_and_value(self, micro_x:Tensor, macro_x:Tensor, pos_type:Tensor, p_value:Tensor, action:int=None, mask_action:Tensor=None):
        actor_logits, value, belief_logits, change_logits = self.forward(micro_x, macro_x, pos_type, p_value)
        if mask_action is not None: actor_logits = actor_logits.masked_fill(~mask_action, -9e8)
        probs = Categorical(logits=actor_logits)
        if action is None: action = probs.sample() 
        log_prob = probs.log_prob(action)
        dist_entropy = probs.entropy()
        belief_prob = torch.softmax(belief_logits, dim=-1)
        change_prob = torch.sigmoid(change_logits)
        #log_prob is the probability action
        #dist_entropy is the entropy Bonus
        #value is the value for critic
        #belief_probs is the probability for belief
        #belief_entropy
        return action, log_prob, dist_entropy, value, belief_logits, change_logits, belief_prob, change_prob, actor_logits

# FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # Alpha permet de garder les poids de classes si on le souhaite
        self.alpha = alpha 

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        # Application de l'équation de la Focal Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss.mean()
