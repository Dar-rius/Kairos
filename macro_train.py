import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from agent.model import MacroHead
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

if __name__ == "__main__":
    DATA_PATH = './data_off/train_test/'
    train_feature_set = pd.read_csv(f"{DATA_PATH}metric_pretrain.csv").iloc[:, 1:]
    train_target_set = pd.read_csv(f"{DATA_PATH}state_pretrain.csv").iloc[:, 1:]
    df_full = pd.merge(train_feature_set, train_target_set, left_index=True, right_index=True)
    #df_test = pd.merge(test_feature_set, test_target_set, left_index=True, right_index=True)
    #df_full = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    df_full["state"] = df_full["state"].shift(-1)
    df_final = df_full.dropna()
    feature_cols = ['mvrv_z_score','nvt_smooth','hashRate_change','log_return','drawdown_micro','rsi_7','rsi_14', 'mvrv_momentum', 'nvt_dynamic', 'rsi_slop']
    X = df_final[feature_cols].values.astype(np.float32)
    y = df_final['state'].values.astype(np.int64)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    weights_tensor = torch.FloatTensor([1.01, 3.5, 4.5])
    lr = 0.0005
    epochs = 150
    batch_size = 64
    tscv = TimeSeriesSplit(n_splits=4)
    MACRO_DIM = train_feature_set.shape[1]
    fold = 0
    all_y_true = []
    all_y_pred = []
    class_names = ['Stable (0)', 'Volatile (1)', 'Crisis (2)']
    model = MacroHead(macro_dim=MACRO_DIM, num_regimes=3)
    for train_index, val_index in tscv.split(X):
        fold += 1
        print(f"\n--- FOLD {fold} ---")
        print(f"Train indices: {len(train_index)} jours | Val indices: {len(val_index)} jours")

        X_train_raw, X_val_raw = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)

        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.LongTensor(y_val)

        criterion = nn.CrossEntropyLoss(weight=weights_tensor) # Gère les logits et les classes (0,1,2)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            _, logits = model(X_train_tensor)
            loss = criterion(logits, y_train_tensor)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            _, val_logits = model(X_val_tensor)
            predictions = torch.argmax(val_logits, dim=1).numpy()
        all_y_true.extend(y_val)
        all_y_pred.extend(predictions)
        # Optionnel : Afficher la précision juste pour ce fold
        fold_acc = np.mean(predictions == y_val)
        print(f"Fold {fold}: Accuracy = {fold_acc:.2%}")
        # G. Métriques
        precision = precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
        recall = recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0)

        print(f"Précision Globale (Weighted): {precision:.4f}")
        print(f"Recall Global (Weighted)   : {recall:.4f}")
        print("\nRapport Détaillé par Classe :")
        print(classification_report(all_y_true, all_y_pred, target_names=class_names, zero_division=0))

        cm = confusion_matrix(all_y_true, all_y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names)
        plt.xlabel('Predit (Ce que le modèle pense)')
        plt.ylabel('Réel (Ce qui est arrivé)')
        plt.title('Matrice de Confusion - Validation Walk-Forward')
        plt.savefig('big_result.png')
    torch.save(model.state_dict(), './agent/save/belief_head.pt')
