import os
import datetime
import torch
from torch import Tensor
import torch.optim as optim
import pandas as pd
import numpy as np
from agent.model import MacroHead, FocalLoss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch import nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader
from collections import deque
import joblib
import optuna

DATA_PATH = './data_off/train_test/'
MAT_CONF_PATH = "./runs/train_macro"
MODEL_PATH = "./agent/save"
EPOCHS = 30          # Nombre d'époques par fold (peut être ajusté)
BATCH_SIZE = 64      # Fixe pour simplifier (peut aussi être optimisé)
NUM_TRIALS = 50      # Nombre d'essais Optuna

# Chargement des données
train_feature_set = pd.read_csv(f"{DATA_PATH}metric_pretrain.csv").iloc[:, 1:]
train_target_belief = pd.read_csv(f"{DATA_PATH}state_pretrain.csv").iloc[:, 1:]
train_target_change = pd.read_csv(f"{DATA_PATH}change_pretrain.csv").iloc[:, 1:]

feature_cols = ['mvrv_z_score','mom_24h','mom_168h','mom_168h_z','hashRate_change',
                'log_return','drawdown_micro', 'vol_garch', 'vol_parkinson',
                'RSI_7','RSI_14', 'mvrv_momentum', 'nvt_momentum', 'rsi_slop',
                'mvrv_lag1', 'mvrv_lag3', 'vol_lag1', 'vol_lag3', 'vol_diff',
                'mvrv_diff', 'hashRate_ma7']

# Fusion et décalage des cibles (prédiction du lendemain)
df_full = pd.merge(train_feature_set, train_target_belief, left_index=True, right_index=True)
df_full = pd.merge(df_full, train_target_change, left_index=True, right_index=True)
df_full["regime"] = df_full["regime"].shift(-1)
df_full["change"] = df_full["change"].shift(-1)
df_final = df_full.dropna()

X = df_final[feature_cols].values.astype(np.float32)
y_belief = df_final['regime'].values.astype(np.int64)
y_change = df_final['change'].values.astype(np.int64)

MACRO_DIM = X.shape[1]

# Validation croisée temporelle
tscv = TimeSeriesSplit(n_splits=10)

def objective(trial):
    # Hyperparamètres à optimiser
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    # Poids pour les 3 classes du régime (classe 0 fixée à 1)
    w1 = trial.suggest_float('belief_w1', 0.5, 5.0)
    w2 = trial.suggest_float('belief_w2', 0.5, 5.0)
    # Poids pour la classe 1 du changement (classe 0 fixée à 1)
    change_w1 = trial.suggest_float('change_w1', 0.5, 5.0)
    # Optionnel : poids pour le terme de régularisation (weight decay)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    belief_weights = torch.FloatTensor([1.0, w1, w2])
    change_weights = torch.FloatTensor([1.0, change_w1])

    # Boucle de validation croisée
    all_belief_f1 = []
    all_change_f1 = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Séparation
        X_train, X_val = X[train_idx], X[val_idx]
        y_train_belief, y_val_belief = y_belief[train_idx], y_belief[val_idx]
        y_train_change, y_val_change = y_change[train_idx], y_change[val_idx]

        # Normalisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.LongTensor(y_train_belief),
            torch.LongTensor(y_train_change)
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_belief_tensor = torch.LongTensor(y_val_belief)
        y_val_change_tensor = torch.LongTensor(y_val_change)

        # Modèle et optimiseur
        model = MacroHead(macro_dim=MACRO_DIM, num_regimes=3, num_changes=2)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Critères
        criterion_belief = FocalLoss(alpha=belief_weights, gamma=2.0)
        criterion_change = FocalLoss(alpha=change_weights, gamma=2.0)

        # Entraînement
        model.train()
        for epoch in range(EPOCHS):
            for batch_x, batch_y_b, batch_y_c in train_loader:
                optimizer.zero_grad()
                _, regime_logits, change_logits = model(batch_x)
                loss_b = criterion_belief(regime_logits, batch_y_b)
                loss_c = criterion_change(change_logits, batch_y_c)
                loss = loss_b + loss_c
                loss.backward()
                optimizer.step()

        # Évaluation sur la validation
        model.eval()
        with torch.no_grad():
            _, regime_logits, change_logits = model(X_val_tensor)
            pred_belief = torch.argmax(regime_logits, dim=1).cpu().numpy()
            pred_change = torch.argmax(change_logits, dim=1).cpu().numpy()

        # F1-score pondéré pour chaque tâche
        f1_b = f1_score(y_val_belief, pred_belief, average='weighted', zero_division=0)
        f1_c = f1_score(y_val_change, pred_change, average='weighted', zero_division=0)
        all_belief_f1.append(f1_b)
        all_change_f1.append(f1_c)

    # Score global : moyenne des F1 pondérés des deux têtes
    avg_f1 = (np.mean(all_belief_f1) + np.mean(all_change_f1)) / 2.0
    return avg_f1

study = optuna.create_study(direction = 'maximize',
                            storage="sqlite:///db.sqlite3",
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=150)
print(study.best_params)
