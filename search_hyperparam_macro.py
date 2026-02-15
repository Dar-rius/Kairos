from tqdm import tqdm
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import optuna
from agent.model import MacroHead
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader

DATA_PATH = './data_off/train_test/'
train_feature_set = pd.read_csv(f"{DATA_PATH}metric_pretrain.csv").iloc[:, 1:]
train_target_set = pd.read_csv(f"{DATA_PATH}state_pretrain.csv").iloc[:, 1:]
MACRO_DIM = train_feature_set.shape[1]
df_full = pd.merge(train_feature_set, train_target_set, left_index=True, right_index=True)
df_full["regime"] = df_full["regime"].shift(-1)
df_final = df_full.dropna()
feature_cols = ['mvrv_z_score','nvt_smooth','hashRate_change','log_return','drawdown_micro', 'volatility',  'vol_park', 'rsi_7','rsi_14', 'mvrv_momentum', 'nvt_dynamic', 'rsi_slop']
X = df_final[feature_cols].values.astype(np.float32)
y = df_final['regime'].values.astype(np.int64)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)

def objective(trial):
# Agent Hyperparam
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    weights_1 = trial.suggest_float("weights_1", 0.1, 10.0)
    weights_2 = trial.suggest_float("weights_2", 0.1, 10.0)
    weights_3 = trial.suggest_float("weights_3", 0.1, 10.0)

    tscv = TimeSeriesSplit(n_splits=5)
    weights_tensor = torch.FloatTensor([weights_1, weights_2, weights_3])

    #Search Best Params
    fold_acc_list: list[float] = []
    for train_index, val_index in tscv.split(X):
        x_train_raw, x_val_raw = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_raw)
        x_val_scaled = scaler.transform(x_val_raw)

        # Création des DataLoaders pour utiliser le batch_size
        train_dataset = TensorDataset(torch.FloatTensor(x_train_scaled), torch.LongTensor(y_train))
        # shuffle=False est crucial en séries temporelles
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        x_val_tensor = torch.FloatTensor(x_val_scaled)
        y_val_tensor = torch.LongTensor(y_val)

        model = MacroHead(macro_dim=MACRO_DIM, num_regimes=3)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for _ in range(200):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                _, logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            _, val_logits = model(x_val_tensor)
            predictions = torch.argmax(val_logits, dim=1).cpu().numpy()
        fold_acc = np.mean(predictions == y_val)
        fold_acc_list.append(fold_acc)
    mean_acc = np.mean(fold_acc_list)
    # For optuna
    trial.report(mean_acc, 1)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    return mean_acc

study = optuna.create_study(direction = 'maximize',
                            storage="sqlite:///db.sqlite3",
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.MedianPruner(),
                            study_name="Macro Head Hyperparams")
study.optimize(objective, n_trials=100)
print(study.best_params)
