import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import Tensor
import numpy as np

class MacroHead(nn.Module):
    def __init__(self, macro_dim:int, num_regimes:int=3):
        super(MacroHead, self).__init__()
        self.macro_net = nn.Sequential(
            nn.Linear(macro_dim, 128),
            nn.LayerNorm(128),     # Stabilise les signaux financiers
            nn.ReLU(),
            nn.Dropout(0.3),         # Désactive 30% des neurones (Anti-Overfit)
            nn.Linear(128, 32),
            nn.LayerNorm(32),      # Stabilise encore
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.belief_head = nn.Linear(32, num_regimes)
        self._init_weights()

    def _init_weights(self):
        for layer in self.macro_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.belief_head.weight, gain=1.0)
        nn.init.constant_(self.belief_head.bias, 0.0)
        

    def forward(self, macro_x:Tensor):
        x = self.macro_net(macro_x)
        belief_logits = self.belief_head(x)
        return x, belief_logits

class Agent(nn.Module):
    def __init__(self, micro_dim:int, action_dim:int, num_regimes:int=3, pretrained_model=None):
        super(Agent, self).__init__()
        self.belief_head = pretrained_model
        self.micro_lstm = nn.LSTM(micro_dim, 128, batch_first=True)
        
        # --- FUSION HIÉRARCHIQUE ---
        # 128 (Micro) + 32 (Macro Context) + 3 (Macro Explicit Prediction)
        fusion_dim = 128 + 32 + num_regimes

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
        #init critic head
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, micro_x:Tensor, macro_x:Tensor):
        # System 2
        with torch.no_grad():
            macro_feat, belief_logits = self.belief_head(macro_x)
        macro_feat = macro_feat.detach()
        belief_logits = belief_logits.detach()
        current_belief_probs = torch.softmax(belief_logits, dim=1)
        # SYSTEM 1
        self.micro_lstm.flatten_parameters()
        _, (h_n, _) = self.micro_lstm(micro_x)
        micro_feat = h_n[-1]
        # FUSION (context)
        context = torch.cat([micro_feat, macro_feat, current_belief_probs], dim=1)
        action_logits = self.actor_layer(context)
        value = self.critic(context)
        return action_logits, value, belief_logits

    def get_action_and_value(self, micro_x:Tensor, macro_x:Tensor, action:int|None=None, mask_action:Tensor=None):
        actor_logits, value, belief_logits = self.forward(micro_x, macro_x)
        if mask_action is not None: actor_logits = actor_logits.masked_fill(~mask_action, -9e8)
        probs = Categorical(logits=actor_logits)
        if action is None: action = probs.sample()
        log_prob = probs.log_prob(action)
        dist_entropy = probs.entropy()
        #belief_probs = F.softmax(belief_logits, dim=1)
        #log_prob is the probability action
        #dist_entropy is the entropy Bonus
        #value is the value for critic
        #belief_probs is the probability for belief
        #belief_entropy
        return action, log_prob, dist_entropy, value, belief_logits

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
        pt = torch.exp(-ce_loss) # Probabilité de la classe correcte
        # Application de l'équation de la Focal Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
