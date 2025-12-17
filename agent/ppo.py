import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOTrainer:
    def __init__(self, model, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, value_coef=0.5, belief_coef=0.5, ent_coef=0.01):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Hyperparamètres PPO
        self.gamma = gamma           # Importance du futur
        self.gae_lambda = gae_lambda # Lissage de l'avantage
        self.clip_eps = clip_eps     # Empêche les changements trop brutaux
        
        # Coefficients de la Loss Totale
        self.value_coef = value_coef
        self.belief_coef = belief_coef
        self.ent_coef = ent_coef
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss() # Pour la classification du régime (Belief)

    def compute_gae(self, rewards, values, masks, next_value):
        """
        On utilise un mask pour empecher l'agent de faire une correlation les steps
        suivant apres la fin du batch
        Calcul de l'Avantage (Generalized Advantage Estimation)
        C'est ce qui dit à l'agent : "Cette action était X fois mieux que prévu."
        """
        returns = []
        gae = 0
        values = values + [next_value]
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + (self.gamma * values[step + 1] * masks[step]) - values[step]
            gae = delta + (self.gamma * self.gae_lambda) * masks[step] * gae
            returns.insert(0, gae + values[step])
            
        return returns

    def update(self, memory, batch_size=64, epochs=4):
        """
        Cœur de l'algorithme PPO : Mise à jour des poids du réseau.
        """
        # 1. Conversion des listes en Tensors
        micro_states = torch.FloatTensor(np.array(memory['micro_states']))
        macro_states = torch.FloatTensor(np.array(memory['macro_states']))
        actions = torch.LongTensor(memory['actions'])
        old_log_probs = torch.FloatTensor(memory['log_probs'])
        returns = torch.FloatTensor(memory['returns'])
        advantages = torch.FloatTensor(memory['advantages'])
        target_regimes = torch.LongTensor(memory['target_regimes']) # Le vrai régime (0,1,2) calculé par l'env

        # Normalisation des avantages (Crucial pour la stabilité)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. Boucle d'optimisation (Plusieurs passages sur les données)
        dataset_size = len(actions)
        indices = np.arange(dataset_size)

        for _ in range(epochs):
            np.random.shuffle(indices) # Mélange pour casser les corrélations
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]

                # A. Ré-évaluation avec le modèle actuel (qui change à chaque step)
                _, new_log_probs, dist_entropy, new_values, belief_logits, _ = self.model.get_action_and_value(
                    micro_states[idx], 
                    macro_states[idx], 
                    actions[idx]
                )
                
                # B. Calcul du Ratio (Probabilité Nouvelle / Probabilité Ancienne)
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                
                # C. Loss PPO (Policy Loss) - Le cœur du "Clip"
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # D. Loss Value (Critic) - MSE
                value_loss = self.mse_loss(new_values.flatten(), returns[idx])
                
                # E. Loss Belief (Auxiliary) - Cross Entropy
                # On compare la prédiction (belief_logits) avec la réalité (target_regimes)
                belief_loss = self.ce_loss(belief_logits, target_regimes[idx])

                # F. Loss Totale (Combinaison pondérée)
                loss = policy_loss + \
                       (self.value_coef * value_loss) + \
                       (self.belief_coef * belief_loss) - \
                       (self.ent_coef * dist_entropy.mean()) # Bonus d'exploration

                # G. Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5) # Sécurité anti-explosion
                self.optimizer.step()

        return loss.item()
