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

DATA_PATH = './data_off/train_test/'
train_feature_set = pd.read_csv(f"{DATA_PATH}metric_pretrain.csv").iloc[:, 1:]
train_target_set = pd.read_csv(f"{DATA_PATH}state_pretrain.csv").iloc[:, 1:]
LR = 0.0008
EPOCHS = 30
BATCH_SIZE = 64
MACRO_DIM = train_feature_set.shape[1]
all_y_true = []
all_y_pred = []
class_names = ['Stable (0)', 'Volatile (1)', 'Crisis (2)']
tscv = TimeSeriesSplit(n_splits=10)

df_full = pd.merge(train_feature_set, train_target_set, left_index=True, right_index=True)
df_full["regime"] = df_full["regime"].shift(-1)
df_final = df_full.dropna()
feature_cols = ['mvrv_z_score','mom_24h','mom_168h','mom_168h_z','hashRate_change','log_return','drawdown_micro', 'volatility',  'vol_park', 'RSI_7','RSI_14', 'mvrv_momentum', 'nvt_dynamic', 'rsi_slop', 'mvrv_lag1', 'mvrv_lag3', 'vol_lag1', 'vol_lag3', 'vol_diff', 'mvrv_diff', 'hashRate_ma7']
X = df_final[feature_cols].values.astype(np.float32)
y = df_final['regime'].values.astype(np.int64)
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

print(f"Précision Globale (Weighted): {precision:.4f}")
print(f"Recall Global (Weighted)   : {recall:.4f}")
print("\nRapport Détaillé par Classe :")
print(classification_report(all_y_true, all_y_pred, target_names=class_names, zero_division=0))

# Plot the confusion matrix
cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Confusion Matrix - Validation Walk-Forward')
plt.savefig('./train_macro/confusion_matrix.png')

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
        optimizer.zero_grad()
        _, logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

torch.save(final_macro_head.state_dict(), './agent/save/belief_head.pt')
print("Model saved")
