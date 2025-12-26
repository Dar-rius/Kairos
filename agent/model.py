import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class Agent(nn.Module):
    def __init__(self, micro_input_dim:int, macro_input_dim:int, action_dim:int, hidden_dim:int=128):
        super(Agent, self).__init__()
        
        # --- Micro Branch---
        self.lstm = nn.LSTM(
            input_size=micro_input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # --- 2. Macro Branch  ---
        self.macro_layer = nn.Sequential(
            nn.Linear(macro_input_dim, 32),
            nn.ReLU())
        
        # --- 3. Fusion ---
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),
            nn.ReLU())
        
        # --- 4. Heads ---
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)
        self.belief = nn.Linear(64, 3)
        self._init_weights()

    def _init_weights(self):
        # Initialize LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=np.sqrt(1))
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        for layer in [self.macro_layer[0], self.shared_layer[0], self.actor, self.critic, self.belief]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(0.01))
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, micro_x:np.array, macro_x:np.array):
        _, (h_n, _) = self.lstm(micro_x)
        micro_out = h_n[-1]
        macro_out = self.macro_layer(macro_x)
        #Concat O_micro and O_macro
        fused = torch.cat([micro_out, macro_out], dim=1)
        latent = self.shared_layer(fused)
        actor_logits = self.actor(latent)
        value = self.critic(latent)
        belief_logits = self.belief(latent)
        return actor_logits, value, belief_logits

    def get_action_and_value(self, micro_x:np.array, macro_x:np.array, action:int=None):
        actor_logits, value, belief_logits = self.forward(micro_x, macro_x)
        probs = Categorical(logits=actor_logits)
        if action is None: action = probs.sample()
        log_prob = probs.log_prob(action)
        dist_entropy = probs.entropy()
        belief_probs = F.softmax(belief_logits, dim=1)
        #log_prob is the probability action
        #dist_entropy is the entropy Bonus
        #value is the value for critic
        #belief_probs is the probability for belief
        return action, log_prob, dist_entropy, value, belief_probs
