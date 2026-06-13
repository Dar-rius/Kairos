# Kairos

Agent de trading Bitcoin basé sur le **Reinforcement Learning** (PPO) avec une architecture双系统 (System 1 / System 2) inspirée du modèle cognitif de Kahneman.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      System 2 (MacroHead)               │
│  Métriques on-chain quotidiennes → Regime + Change      │
│  (MVRV, NVT, Hash Rate, RSI, etc.)                     │
│  Entraîné séparément avec FocalLoss                     │
└───────────────────────┬─────────────────────────────────┘
                        │ belief_probs + change_probs
┌───────────────────────▼─────────────────────────────────┐
│                      System 1 (Agent PPO)               │
│  Micro LSTM (fenêtre 24h) ─┐                           │
│  Macro context ─────────────┤                           │
│  Position actuelle ────────┼─→ Fusion → Actor + Critic │
│  Valeur portfolio ─────────┘   (137 features)           │
│                                                         │
│  Actions : Short (0) / Hold (1) / Long (2)             │
└─────────────────────────────────────────────────────────┘
```

- **System 2** (`MacroHead`) : Classifie le régime de marché (Stable/Volatile/Crisis) et prédit les changements. Pré-entraîné sur des données 2015-2018.
- **System 1** (`Agent`) : Agent PPO qui fusionne les features micro (données horaires via LSTM) avec le contexte macro pour décider de l'action suivante.

## Prérequis

- **Python >= 3.13**
- **uv** (gestionnaire de packages, pas pip)
- **CUDA** (optionnel, pour l'entraînement GPU)

## Installation

### 1. Cloner le repository

```bash
git clone git@github.com:Dar-rius/Kairos.git
cd Kairos
```

### 2. Installer uv

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Installer les dépendances

```bash
# Installer PyTorch (CUDA) + toutes les dépendances
uv sync

# Installer les outils de développement
uv pip install pytest mypy --extra-index-url https://pypi.org/simple
```

> **Note** : Le index par défaut est configuré sur `https://download.pytorch.org/whl/cu130` dans `pyproject.toml`. Les packages non-PyTorch sont installés depuis PyPI via `--extra-index-url`.

### 4. Vérifier l'installation

```bash
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Données

Les données ne sont pas incluses dans le repository. Elles doivent être placées dans `data_off/train_test/` :

```
data_off/train_test/
├── price_train.csv          # Données horaires d'entraînement (2018-2023)
├── price_test.csv           # Données horaires de test (2023-2026)
├── metric_pretrain.csv      # Métriques macro pretrain (2015-2018)
├── metric_train.csv         # Métriques macro entraînement (2018-2023)
├── metric_test.csv          # Métriques macro test (2023-2026)
├── state_train.csv          # Labels de regime (entraînement)
├── state_test.csv           # Labels de regime (test)
├── change_train.csv         # Labels de changement (entraînement)
├── change_test.csv          # Labels de changement (test)
├── price_close_train.csv    # Prix de clôture (entraînement)
└── price_close_test.csv     # Prix de clôture (test)
```

Pour préprocesser les données brutes en CSV séparés :

```bash
uv run python script.py
```

## Pipeline d'entraînement

### Étape 1 : Entraîner MacroHead (System 2)

```bash
uv run python train_macro_head.py
```

**Sorties** :
- `agent/save/macro_head.pt` — Poids du modèle
- `agent/save/macro_scaler.pkl` — Scaler StandardScaler pour les features macro
- `runs/train_macro/` — Matrices de confusion

### Étape 2 : Entraîner l'Agent (System 1)

```bash
uv run python train_agent.py
```

**Prérequis** : Connexion `wandb` (logging des métriques).

**Sorties** :
- `agent/save/agent_saved.pt` — Poids de l'agent
- `agent/save/macro_head_1.pt` — Copie du MacroHead après fine-tuning

> **Note** : L'entraînement est forcé sur CPU malgré la config GPU. Le batch size est de 64 et le rollout fait 2048 steps.

### Étape 3 : Backtest

```bash
uv run python test_agent.py
```

**Sorties** :
- `runs/test/backtest_result_{timestamp}.png` — Graphique du backtest

## Hyperparameter Search

### MacroHead (System 2)

```bash
uv run python train_macro_hyperparams.py
```

Utilise Optuna (50 trials, 150 max trials) avec validation croisée temporelle (10 folds).

### Agent (System 1)

```bash
uv run python search_hyperparams_agent.py
```

Utilise Optuna (50 trials, GPU si disponible). Paramètres recherchés :
- `lr` : learning rate (1e-6 à 1e-3)
- `gamma` : discount factor (0.80 à 0.99)
- `gae_lambda` : GAE lambda (0.80 à 0.99)
- `ent_coef` : coefficient d'entropie (0.01 à 0.9)
- `value_coef` : coefficient de la loss value (0.05 à 0.5)
- `belief_coef` : coefficient de la loss belief (0.01 à 0.5)
- `batch_size` : {64, 128, 256}

## Tests

```bash
uv run pytest tests/
```

> **Attention** : Les tests nécessitent `data_off/unit_test/` qui n'est pas inclus. Ils échoueront sans ce répertoire.

## Type Checking

```bash
mypy kairos/
```

> **Note** : `mypy.ini` référence un dossier `rl_trade` inexistant. Utilisez la commande ci-dessus directement.

## Docker

```bash
docker build -t kairos .
docker run --gpus all kairos
```

L'image utilise le conteneur NVIDIA PyTorch (`nvcr.io/nvidia/pytorch:25.08-py3`).

## Structure du projet

```
Kairos/
├── agent/
│   ├── model.py              # MacroHead, Agent, FocalLoss
│   ├── ppo_belief.py         # Algorithme PPO avec losses auxiliaires
│   ├── buffer.py             # Rollout buffer (tensors pré-alloués)
│   ├── save/                 # Poids sauvegardés (gitignoré)
│   └── __init__.py
├── kairos/
│   ├── env.py                # Environnement de trading
│   ├── compute.py            # Reward DSR, Sharpe, max drawdown
│   ├── processing.py         # Conversions unitaires
│   └── __init__.py
├── tests/
│   └── test_env.py           # Tests unitaires (nécessite données)
├── data_off/
│   └── train_test/           # Jeux de données (gitignoré)
├── runs/
│   └── test/                 # Graphiques de backtest
├── train_macro_head.py       # Entraînement System 2
├── train_agent.py            # Entraînement System 1
├── test_agent.py             # Backtest
├── search_hyperparams_agent.py    # Optuna pour l'Agent
├── train_macro_hyperparams.py     # Optuna pour MacroHead
├── visualizer.py             # Visualisations wandb (3D scatter)
├── script.py                 # Préprocessing des données
├── pyproject.toml            # Configuration uv
├── requirements.txt          # Dépendances complètes
├── dockerfile                # Image Docker NVIDIA
└── AGENTS.md                 # Documentation technique
```

## Fonctionnalités clés

- **Reward DSR-based** : Le reward est basé sur le Deflated Sharpe Ratio avec une période de warmup de 10 steps
- **Action masking** : Possibilité de masquer certaines actions (protection liquidation)
- **FocalLoss** : Gestion du déséquilibre de classes pour les prédictions macro
- **PPO clipping** : Stabilisation de l'entraînement avec clip epsilon configurable
- **Gradient clipping** : Max norm 1.0 pour éviter les explosions de gradients
- **GAE** : Generalized Advantage Estimation pour l'estimation des avantages
- **Learning rate decay** : Décroissance linéaire du LR sur toute la durée d'entraînement

## Auteur

**Dar-rius** — mohamedtine17@gmail.com
