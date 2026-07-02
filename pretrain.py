import os
import datetime
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch import Tensor
from agent.model import MacroHead, FocalLoss
from torch import nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader
from collections import deque


#Generate a heatmap of confusion matrix 
def gen_conf_matrix(y_true: np.array, y_pred: np.array, path:str, class_names:list):
    # confusion matrix graphic
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predictions')
    plt.ylabel('Reality')
    plt.title('Confusion Matrix - Validation Walk-Forward')

    #save plot
    if not os.path.exists(path): os.makedirs(path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{path}/confusion_matrix_{timestamp}.png")
    print(f"Confusion Matrix saved: {path}/confusion_matrix_{timestamp}.png")

#Generate metric about the model predictions performance
def gen_metric(y_true:deque[int], y_pred:deque[int], class_names:list):
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Précision Globale (Weighted): {precision:.4f}")
    print(f"Recall Global (Weighted)   : {recall:.4f}")
    print(f"F1-Score Global (Weighted)   : {f1:.4f}")
    print("\nRapport Détaillé par Classe :")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


#Config
DEVICE = 'cuda:0'
DATA_PATH = './data_off/train_test/'
train_feature_set = pd.read_csv(f"{DATA_PATH}metric_pretrain.csv").iloc[:, 1:]
train_target_belief = pd.read_csv(f"{DATA_PATH}regime_pretrain.csv").iloc[:, 1:]
train_target_change = pd.read_csv(f"{DATA_PATH}change_pretrain.csv").iloc[:, 1:]
MAT_CONF_PATH = "./runs/train_macro"
MODEL_PATH = "./agent/save"

#Training variables
LR = 0.003850091902319146
EPOCHS = 30
BATCH_SIZE = 64
MACRO_DIM = train_feature_set.shape[1]

#The train historic
all_y_belief_true : deque[int] = deque()
all_y_belief_pred : deque[int] = deque()
all_y_change_true : deque[int] = deque()
all_y_change_pred : deque[int] = deque()

# Named all class for 2 targets (regime and change)
class_names_belief = ['Stable (0)', 'Volatile (1)', 'Crisis (2)']
class_names_change = ['NO CHANGE (0)', 'CHANGE (1)']
tscv = TimeSeriesSplit(n_splits=10)

#Merge the target of all auxilliary task with the main dataset
df_full = pd.merge(train_feature_set, train_target_belief, left_index=True, right_index=True)
df_full = pd.merge(df_full, train_target_change , left_index=True, right_index=True)

#Shift values up by one row to get  the t+1 step
df_full["regime"] = df_full["regime"].shift(-1)
df_full["change"] = df_full["change"].shift(-1)
df_final = df_full.dropna()

# Select all features
feature_cols = ['mvrv_z_score','mom_24h','mom_168h',
                'mom_168h_z','hashRate_change','log_return',
                'drawdown_micro', 'vol_garch',  'vol_parkinson',
                'RSI_7','RSI_14', 'mvrv_momentum', 'nvt_momentum',
                'rsi_slop', 'mvrv_lag1', 'mvrv_lag3',
                'vol_lag1', 'vol_lag3', 'vol_diff',
                'mvrv_diff', 'hashRate_ma7']
X = df_final[feature_cols].values.astype(np.float32)

#Select target of belief and change
y_belief = df_final['regime'].values.astype(np.int64)
y_change = df_final['change'].values.astype(np.int64)
fold = 0

#Weighted the class of each class
weights_tensor_belief = torch.FloatTensor([1., 0.8835228721407905, 0.9362682099573649])
weights_tensor_change = torch.FloatTensor([1., 0.981713583978431])

#Pipeline Train
for train_index, val_index in tscv.split(X):
    fold += 1
    print(f"\n--- FOLD {fold} ---")
    print(f"Train indices: {len(train_index)} jours | Val indices: {len(val_index)} jours")

    X_train, X_val = X[train_index], X[val_index]
    y_train_belief, y_val_belief = y_belief[train_index], y_belief[val_index]
    y_train_change, y_val_change = y_change[train_index], y_change[val_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    #Convert dataset to Tensor
    train_tensor = TensorDataset(torch.FloatTensor(X_train_scaled), 
                                 torch.LongTensor(y_train_belief), 
                                 torch.LongTensor(y_train_change))
    train_loader = DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=True)
    x_val_tensor = torch.FloatTensor(X_val_scaled)

    #Initialize the errors evaluator
    criterion_belief = FocalLoss(alpha=weights_tensor_belief)
    criterion_change = FocalLoss(alpha=weights_tensor_change)
    
    model = MacroHead(macro_dim=MACRO_DIM, num_regimes=3, num_changes=2)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=6.734119081882703e-06)
    
    #Start train model
    model.train()
    for epoch in range(EPOCHS):
        for batch_x, batch_y_belief, batch_y_change in train_loader:
            optimizer.zero_grad()
            _, belief_logit, change_logit = model(batch_x)
            belief_loss = criterion_belief(belief_logit, batch_y_belief)
            change_loss = criterion_change(change_logit, batch_y_change)

            global_loss = belief_loss + change_loss
            global_loss.backward()
            optimizer.step()

    #Evaluate the model validation phase
    model.eval()
    with torch.no_grad():
        _, val_belief_logits, val_change_logits = model(x_val_tensor)
        pred_belief = torch.argmax(val_belief_logits, dim=1).cpu().numpy()
        pred_change = torch.argmax(val_change_logits, dim=1).cpu().numpy()

    #Store the metrics
    all_y_belief_true.extend(y_val_belief)
    all_y_change_true.extend(y_val_change)
    all_y_belief_pred.extend(pred_belief)
    all_y_change_pred.extend(pred_change)
    fold_acc_belief = np.mean(pred_belief == y_val_belief)
    fold_acc_change = np.mean(pred_change == y_val_change)

    print(f"Fold {fold} belief: Accuracy = {fold_acc_belief:.2%}")
    print(f"Fold {fold} change: Accuracy = {fold_acc_change:.2%}")

#Belief_head's Metrics
gen_metric(all_y_belief_true, all_y_belief_pred, class_names_belief)
#Change_head's Metrics
gen_metric(all_y_change_true, all_y_change_pred, class_names_change)

# Plot the confusion matrix belief
gen_conf_matrix(all_y_belief_true, all_y_belief_pred, f"{MAT_CONF_PATH}/belief", class_names_belief)
# Plot the confusion matrix change
gen_conf_matrix(all_y_change_true, all_y_change_pred, f"{MAT_CONF_PATH}/change", class_names_change)

# Last model train
scaler = StandardScaler()
x_full_final = scaler.fit_transform(X)
train_tensor_final = TensorDataset(torch.FloatTensor(x_full_final), torch.LongTensor(y_belief), torch.LongTensor(y_change))
train_loader_final = DataLoader(train_tensor_final, batch_size=BATCH_SIZE, shuffle=True)

final_macro_head = MacroHead(macro_dim=MACRO_DIM, num_regimes=3, num_changes=2)
final_optimizer = optim.Adam(final_macro_head.parameters(), lr=LR)

#Start train
final_macro_head.train()
for epoch in range(EPOCHS):
    for batch_x, batch_y_belief, batch_y_change in train_loader_final:
        final_optimizer.zero_grad()
        _, logit_belief, logit_change = final_macro_head(batch_x)
        loss_belief = criterion_belief(logit_belief, batch_y_belief)
        loss_change = criterion_change(logit_change, batch_y_change)
        global_loss = loss_belief + loss_change
        global_loss.backward()
        final_optimizer.step()

if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)
# Save model
torch.save(final_macro_head.state_dict(), f'{MODEL_PATH}/macro_head.pt')
print(f"Model is saved in: {MODEL_PATH}/macro_head.pt")

joblib.dump(scaler, f'{MODEL_PATH}/macro_scaler.pkl')
print(f"Scaler is saved: {MODEL_PATH}/macro_scaler.pkl")
