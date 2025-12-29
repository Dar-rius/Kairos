import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rl_trade.env import Env
from agent.model import Agent
from tqdm import tqdm

# --- CONFIGURATION ---
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_PATH = './agent/save/agent_saved.pt'  # Chemin vers ton modèle entraîné
DATA_PATH = './data_off/train_test/'

# On veut tester sur tout le dataset ou une partie spécifique (ex: test set)
# Ici, on recharge les mêmes données, mais l'env sera configuré pour tout parcourir.
hour_df = pd.read_csv(f"{DATA_PATH}norm_price.csv").iloc[:, 1:]
macro_df = pd.read_csv(f"{DATA_PATH}metric.csv").iloc[:, 1:]
price_series = pd.read_csv(f"{DATA_PATH}price_close.csv")["Close"]

# --- INITIALISATION ---
# On met ROLLOUT_STEPS très grand pour éviter les resets intempestifs pendant le test
# On veut voir la performance sur une longue période continue.
TEST_STEPS = len(hour_df) - 100

env = Env(hour_df, macro_df, price_series)
ACTION_DIM = env.action_space
STATE_DIM = env.observation_space

# Création de l'agent et chargement des poids
agent = Agent(STATE_DIM[0], STATE_DIM[1], ACTION_DIM).to(DEVICE)
print(f"Load model from {MODEL_PATH}...")
agent.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
agent.eval() # IMPORTANT : Met le modèle en mode évaluation (désactive Dropout, etc.)

# --- BOUCLE DE TEST ---
print("Run the Backtest...")
micro_obs, macro_obs = env.reset()

# Variables pour le tracking
portfolio_history = []
price_history = []
actions_history = []
pnl_history = []

done = False
truncate = False

# On utilise tqdm pour voir la progression
for _ in tqdm(range(TEST_STEPS)):
    
    # 1. Préparation des données (Comme dans train, mais sans gradient)
    micro_t = torch.tensor(micro_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    macro_t = torch.tensor(macro_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    action_mask = env.get_action_mask()
    
    with torch.no_grad():
        # On demande l'action à l'agent
        # Note: En test, on peut vouloir être déterministe (argmax) ou garder le sampling.
        # Avec PPO, le sampling reste souvent utilisé, mais pour un backtest strict,
        # on préfère souvent prendre l'action la plus probable.
        # Ici on garde ton get_action_and_value qui sample, mais comme l'entropie a baissé,
        # il devrait être confiant.
        action_t, _, _, _, _, _ = agent.get_action_and_value(micro_t, macro_t, mask_action=action_mask)
    
    action = action_t.item()
    
    # 2. Step Environment
    # On passe 0 pour belief_entropy car on ne s'entraîne pas
    next_obs, reward, _, truncate, done = env.step(action, entropy_b=0)
    
    # 3. Enregistrement des datas pour l'analyse
    current_val = env.calcul_portfolio_value()
    current_price = env.btc_values[-1] if len(env.btc_values) > 0 else 0
    
    portfolio_history.append(current_val)
    price_history.append(current_price)
    actions_history.append(action) # 0: Hold, 1: Buy, 2: Sell
    pnl_history.append(env.get_pnl())

    if done or truncate:break
    micro_obs, macro_obs = next_obs

# --- VISUALISATION DES RÉSULTATS ---
print(f"Portfolio Final: {portfolio_history[-1]:.2f} $")
print(f"PnL Final (Net): {pnl_history[-1]:.2f} $")

# Création du graphique
plt.figure(figsize=(15, 10))

# Sous-graphique 1 : Prix BTC et Actions
plt.subplot(2, 1, 1)
plt.plot(price_history, label='BTC Price', color='gray', alpha=0.5)

# On récupère les indices où on a acheté (1) et vendu (2)
buy_idx = [i for i, x in enumerate(actions_history) if x == 1]
sell_idx = [i for i, x in enumerate(actions_history) if x == 2]

# Affichage des points d'achat/vente
plt.scatter(buy_idx, [price_history[i] for i in buy_idx], marker='^', color='green', label='Buy', s=50)
plt.scatter(sell_idx, [price_history[i] for i in sell_idx], marker='v', color='red', label='Sell', s=50)

plt.title('Stratégie de Trading (Prix BTC)')
plt.legend()
plt.grid(True)

# Sous-graphique 2 : Valeur du Portefeuille
plt.subplot(2, 1, 2)
plt.plot(portfolio_history, label='Portfolio Value ($)', color='blue')
plt.axhline(y=100000, color='r', linestyle='--', label='Initial Capital') # Assumant 100k départ
plt.title('Evolution du Portefeuille')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('backtest_result.png')
print("Graphique sauvegardé sous 'backtest_result.png'")
plt.show()
