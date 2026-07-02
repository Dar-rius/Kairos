import optuna
import torch
import pandas as pd
import numpy as np
import wandb
from torch import optim
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from optuna.integration.wandb import WeightsAndBiasesCallback


#Config
DATA_PATH = './data_off/train_test/'
MAT_CONF_PATH = "./runs/train_macro"
MODEL_PATH = "./agent/save"
EPOCHS = 30         
BATCH_SIZE = 64
NUM_TRIALS = 50

# Chargement des données
train_feature_set = pd.read_csv(f"{DATA_PATH}metric_pretrain.csv").iloc[:, 1:]
train_target_regime = pd.read_csv(f"{DATA_PATH}regime_pretrain.csv").iloc[:, 1:]
train_target_change = pd.read_csv(f"{DATA_PATH}change_pretrain.csv").iloc[:, 1:]


df_full = pd.merge(train_feature_set, train_target_regime, left_index=True, right_index=True)
df_full = pd.merge(df_full, train_target_change , left_index=True, right_index=True)

#Shift values up by one row to get  the t+1 step
df_full["regime"] = df_full["regime"].shift(-1)
df_full["change"] = df_full["change"].shift(-1)
df_final = df_full.dropna()

# Select all features
feature_cols = ['mvrv_z_score','mom_24h','mom_168h','mom_168h_z','hashRate_change',
                'log_return','drawdown_micro', 'vol_garch', 'vol_parkinson',
                'RSI_7','RSI_14', 'mvrv_momentum', 'nvt_momentum', 'rsi_slop',
                'mvrv_lag1', 'mvrv_lag3', 'vol_lag1', 'vol_lag3', 'vol_diff',
                'mvrv_diff', 'hashRate_ma7']
X = df_final[feature_cols].values.astype(np.float32)

#Select target of belief and change
y_belief = df_final['regime'].values.astype(np.int64)
y_change = df_final['change'].values.astype(np.int64)

MACRO_DIM = X.shape[1]

# Validation croisée temporelle
tscv = TimeSeriesSplit(n_splits=10)

def objective(trial):
    # Train hyper-params
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    # Belief data weight 2 (volatily)
    belief_w1 = trial.suggest_float('belief_w1', 0.5, 5.0)
    # Belief data weight 3 (crisis)
    belief_w2 = trial.suggest_float('belief_w2', 0.5, 5.0)
    # Change data weight 2 (yes is changing)
    change_w1 = trial.suggest_float('change_w1', 0.5, 5.0)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    belief_weights = torch.FloatTensor([1.0, belief_w1, belief_w2])
    change_weights = torch.FloatTensor([1.0, change_w1])

    all_belief_f1 = []
    all_change_f1 = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Select data between  train/val
        X_train, X_val = X[train_idx], X[val_idx]
        y_train_belief, y_val_belief = y_belief[train_idx], y_belief[val_idx]
        y_train_change, y_val_change = y_change[train_idx], y_change[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Convert dataset to tensor
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.LongTensor(y_train_belief),
            torch.LongTensor(y_train_change)
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        x_val_tensor = torch.FloatTensor(X_val_scaled)

        # Initialize class model
        model = MacroHead(macro_dim=MACRO_DIM, num_regimes=3, num_changes=2)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        criterion_belief = FocalLoss(alpha=belief_weights, gamma=2.0)
        criterion_change = FocalLoss(alpha=change_weights, gamma=2.0)

        # Start Training
        model.train()
        for _ in range(EPOCHS):
            for batch_x, batch_y_b, batch_y_c in train_loader:
                optimizer.zero_grad()
                _, regime_logits, change_logits = model(batch_x)
                loss_b = criterion_belief(regime_logits, batch_y_b)
                loss_c = criterion_change(change_logits, batch_y_c)
                loss = loss_b + loss_c
                loss.backward()
                optimizer.step()

        # Evaluate model
        model.eval()
        with torch.no_grad():
            _, regime_logits, change_logits = model(x_val_tensor)
            pred_belief = torch.argmax(regime_logits, dim=1).cpu().numpy()
            pred_change = torch.argmax(change_logits, dim=1).cpu().numpy()

        # calcul the prediction's precision
        f1_b = f1_score(y_val_belief, pred_belief, average='weighted', zero_division=0)
        f1_c = f1_score(y_val_change, pred_change, average='weighted', zero_division=0)
        all_belief_f1.append(f1_b)
        all_change_f1.append(f1_c)

    # Global Score
    avg_f1 = (np.mean(all_belief_f1) + np.mean(all_change_f1)) / 2.0
    return avg_f1

#Initialize the WandbCallback
wandb_kwargs = {
        "project": "kairos",
        "name": "search_hyperparam_macro"
                }
wandbc = WeightsAndBiasesCallback(metric_name="f1_score", wandb_kwargs=wandb_kwargs)

#Create a new dashboard in optuna dashboard 
study = optuna.create_study(direction = 'maximize',
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100, callbacks=[wandbc])
wandb.finish()
print(study.best_params)
