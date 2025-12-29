import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class Agent(nn.Module):
    def __init__(self, micro_dim:int, macro_dim:int, action_dim:int, num_regimes:int=3):
        super(Agent, self).__init__()
       # --- SYSTEM 2 (Le Stratège - Macro) ---
        self.macro_net = nn.Sequential(
            nn.Linear(macro_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32) # Sort un vecteur de contexte latent
        )
        # Tête de prédiction explicite
        self.belief_head = nn.Linear(32, num_regimes)

        # --- SYSTEM 1 (L'Exécutant - Micro) ---
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

        for module in [self.macro_net, self.actor_layer]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)

        for layer in [self.critic, self.belief_head]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.constant_(layer.bias, 0.0)

        actor_out = self.actor_layer[-1]
        nn.init.orthogonal_(actor_out.weight, gain=0.01)
        nn.init.constant_(actor_out.bias, 0.0)

    def forward(self, micro_x:np.array, macro_x:np.array):
        # System 2
        macro_feat = self.macro_net(macro_x)
        belief_logits = self.belief_head(macro_feat)
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

    def get_action_and_value(self, micro_x:np.array, macro_x:np.array, action:int=None, mask_action:np.array=None):
        actor_logits, value, belief_logits = self.forward(micro_x, macro_x)
        if mask_action is not None: actor_logits.masked_fill(~mask_action, -1e8)
        probs = Categorical(logits=actor_logits)
        if action is None: action = probs.sample()
        log_prob = probs.log_prob(action)
        dist_entropy = probs.entropy()
        belief_probs = F.softmax(belief_logits, dim=1)
        belief_entropy = -torch.sum(belief_probs * torch.log(belief_probs + 1e-8), dim=1)
        #log_prob is the probability action
        #dist_entropy is the entropy Bonus
        #value is the value for critic
        #belief_probs is the probability for belief
        #belief_entropy
        return action, log_prob, dist_entropy, value, belief_logits, belief_entropy
