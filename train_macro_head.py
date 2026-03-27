import os
import datetime
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from agent.model import MacroHead, FocalLoss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader
from collections import deque
import joblib

DATA_PATH = './data_off/train_test/'
train_feature_set = pd.read_csv(f"{DATA_PATH}metric_pretrain.csv").iloc[:, 1:]
train_target_set = pd.read_csv(f"{DATA_PATH}state_pretrain.csv").iloc[:, 1:]
MAT_CONF_PATH = "./runs/train_macro"
MODEL_PATH = "./agent/save"
LR = 0.0008
EPOCHS = 30
BATCH_SIZE = 64
MACRO_DIM = train_feature_set.shape[1]
all_y_true : deque[int] = deque()
all_y_pred : deque[int] = deque()
class_names = ['Stable (0)', 'Volatile (1)', 'Crisis (2)']
tscv = TimeSeriesSplit(n_splits=10)

df_full = pd.merge(train_feature_set, train_target_set, left_index=True, right_index=True)
df_full["regime"] = df_full["regime"].shift(-1)
df_final = df_full.dropna()
feature_cols = ['mvrv_z_score','mom_24h','mom_168h','mom_168h_z','hashRate_change','log_return','drawdown_micro', 'vol_garch',  'vol_parkinson', 'RSI_7','RSI_14', 'mvrv_momentum', 'nvt_momentum', 'rsi_slop', 'mvrv_lag1', 'mvrv_lag3', 'vol_lag1', 'vol_lag3', 'vol_diff', 'mvrv_diff', 'hashRate_ma7']
X = df_final[feature_cols].values.astype(np.float32)
y = df_final['regime'].values.astype(np.int64)
print(train_feature_set.isna().sum())
print(train_target_set.isna().sum())
print(np.isinf(X).sum())
print(np.isinf(y).sum())
fold = 0
weights_tensor = torch.FloatTensor([1., 1.3, 3.])

for train_index, val_index in tscv.split(X):
    fold += 1
    print(f"\n--- FOLD {fold} ---")
    print(f"Train indices: {len(train_index)} jours | Val indices: {len(val_index)} jours")

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_tensor = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
    train_loader = DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=True)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    #weights_tensor = torch.FloatTensor(class_weights)
    #print(weights_tensor)
    criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)
    model = MacroHead(macro_dim=MACRO_DIM, num_regimes=3)
    #criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    model.train()
    for epoch in range(EPOCHS):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            _, logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        _, val_logits = model(X_val_tensor)
        predictions = torch.argmax(val_logits, dim=1).cpu().numpy()

    all_y_true.extend(y_val)
    all_y_pred.extend(predictions)
    fold_acc = np.mean(predictions == y_val)
    print(f"Fold {fold}: Accuracy = {fold_acc:.2%}")

# Metrics
precision = precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
recall = recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0)

# Plot the confusion matrix
cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Confusion Matrix - Validation Walk-Forward')
if not os.path.exists(MAT_CONF_PATH): os.makedirs(MAT_CONF_PATH)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"{MAT_CONF_PATH}/confusion_matrix_{timestamp}.png")
print(f"Confusion Matrix saved: {MAT_CONF_PATH}/confusion_matrix_{timestamp}.png")

#Train the finale model and saved it
scaler = StandardScaler()
x_full_final = scaler.fit_transform(X)
train_tensor_final = TensorDataset(torch.FloatTensor(x_full_final), torch.LongTensor(y))
train_loader_final = DataLoader(train_tensor_final, batch_size=BATCH_SIZE, shuffle=True)

final_macro_head = MacroHead(macro_dim=MACRO_DIM, num_regimes=3)
final_optimizer = optim.Adam(final_macro_head.parameters(), lr=LR)
final_macro_head.train()
for epoch in range(EPOCHS):
    for batch_x, batch_y in train_loader_final:
        final_optimizer.zero_grad()
        _, logits = final_macro_head(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        final_optimizer.step()


if not os.path.exists(MODEL_PATH): 
    os.makedirs(MODEL_PATH)
    
# 1. Sauvegarde du modèle réparé
torch.save(final_macro_head.state_dict(), f'{MODEL_PATH}/belief_head.pt')
print(f"Model is saved in: {MODEL_PATH}/belief_head.pt")

joblib.dump(scaler, f'{MODEL_PATH}/macro_scaler.pkl')
print(f"✅ Scaler is saved: {MODEL_PATH}/macro_scaler.pkl")
