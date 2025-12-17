import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class BitcoinAgent(nn.Module):
    def __init__(self, micro_input_dim=6, macro_input_dim=7, action_dim=3, hidden_dim=128):
        super(BitcoinAgent, self).__init__()
        
        # --- 1. BRANCHE MICRO (Technique) AVEC LSTM ---
        # Changement ici : nn.LSTM au lieu de nn.GRU
        self.lstm = nn.LSTM(
            input_size=micro_input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # --- 2. BRANCHE MACRO (Fondamentale) ---
        self.macro_layer = nn.Sequential(
            nn.Linear(macro_input_dim, 32),
            nn.ReLU()
        )
        
        # --- 3. FUSION ---
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),
            nn.ReLU()
        )
        
        # --- 4. TÊTES ---
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)
        self.belief = nn.Linear(64, 3)
        
        self._init_weights()

    def _init_weights(self):
        # On initialise aussi le LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                
        for layer in [self.macro_layer[0], self.shared_layer[0], self.actor, self.critic, self.belief]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, micro_x, macro_x):
        """
        Adaptation pour LSTM : Gestion du tuple (h_n, c_n)
        """
        # 1. Traitement Micro (LSTM)
        # LSTM renvoie : output, (h_n, c_n)
        # h_n : État caché (Short-term memory) -> C'est ce qu'on veut pour la décision
        # c_n : État de cellule (Long-term memory) -> On l'ignore pour la fusion
        
        _, (h_n, _) = self.lstm(micro_x)
        
        # h_n est de forme (num_layers, batch, hidden_size)
        # On prend la dernière couche [-1]
        micro_out = h_n[-1] # Shape: (Batch, 128)
        
        # 2. Traitement Macro
        macro_out = self.macro_layer(macro_x) # Shape: (Batch, 32)
        
        # 3. Fusion
        fused = torch.cat([micro_out, macro_out], dim=1)
        latent = self.shared_layer(fused)
        
        # 4. Sorties
        actor_logits = self.actor(latent)
        value = self.critic(latent)
        belief_logits = self.belief(latent)
        
        return actor_logits, value, belief_logits

    def get_action_and_value(self, micro_x, macro_x, action=None):
        # Cette partie reste identique à la version GRU
        actor_logits, value, belief_logits = self.forward(micro_x, macro_x)
        probs = Categorical(logits=actor_logits)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        dist_entropy = probs.entropy()
        belief_probs = F.softmax(belief_logits, dim=1)
        belief_entropy = -torch.sum(belief_probs * torch.log(belief_probs + 1e-8), dim=1)
        
        return action, log_prob, dist_entropy, value, belief_logits, belief_entropy
