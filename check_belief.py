import matplotlib.pyplot as plt
import seaborn as sns
import torch

state_dict = torch.load("./agent/save/belief_head.pt", weights_only=True)
for name, tensor in state_dict.items():
    print(f"Couche : {name:<30} | Dimension : {list(tensor.shape)}")
    
    # Compte les NaNs et les Infinis
    nb_nans = torch.isnan(tensor).sum().item()
    nb_infs = torch.isinf(tensor).sum().item()

    if nb_nans > 0: print("NaN values are found")
    elif nb_infs > 0: print("Inf values are found") 

poids_couche_1 = state_dict['macro_net.0.weight'].cpu().numpy()

plt.figure(figsize=(10, 6))
sns.heatmap(poids_couche_1, cmap="coolwarm", center=0, cbar=True)
plt.title("Visualisation de la Matrice des Poids (Couche 1)")
plt.xlabel("Neurones d'entrée (Features Macro)")
plt.ylabel("Neurones de sortie (128)")
plt.savefig("data_off/matrice_poids.png")
