> **Note:** A paper about this project will be available soon.

This repository is an experimentation of a new variant of PPO (Proximal Policy Optimisation) called PPO-Belief. In this work, we introduce a new auxiliary task “Belief”, it learns to predict the hidden state in a partially observable environment. Also we can see if this architecture can perform in an environment with a lot of hidden state like financial market or physics.

We evaluate this algorithm on the financial market for trading Bitcoin. We chose to trade Bitcoin, because it is known as the asset with the most volatility. We want to see if our algorithm outperforms PPO in this environment.

## Architecture

![Agent architecture](./images/architecture.png)

We introduce system-1 and system-2 architecture. The system-2 takes as input a set of macro states market, it predicts the regime and change in market state for next time step. The system-1 takes as input a set of micro states markets, the latent vector and predictions from system-2 to select actions in the environment (buy, hold, short).

- **System 2** (`MacroHead`): Classifies market regime (Stable/Volatile/Crisis) and predicts imminent changes. Pre-trained on 2015-2018 data.
- **System 1** (`Agent`): PPO agent that fuses micro features (hourly data via LSTM) with macro context to decide the next action.

## Prerequisites

- **Python >= 3.13**
- **uv** (package manager, not pip)
- **CUDA** (optional, for GPU training)

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:Dar-rius/Kairos.git
cd Kairos
```

### 2. Install uv

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Install dependencies

```bash
# Install PyTorch (CUDA) + all dependencies
uv sync

# Install development tools
uv pip install pytest mypy --extra-index-url https://pypi.org/simple
```

> **Note**: The default index is configured to `https://download.pytorch.org/whl/cu130` in `pyproject.toml`. Non-PyTorch packages are installed from PyPI via `--extra-index-url`.

### 4. Verify installation

```bash
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Data

Data is not included in the repository, you can download data from Hugging Face [here](https://huggingface.co/datasets/Darrius2020/RL_quant_btc).

Place your CSV files in `data_off/train_test/`:

```
data_off/train_test/
├── price_train.csv          # Hourly training data (2018-2023)
├── price_test.csv           # Hourly test data (2023-2026)
├── metric_pretrain.csv      # Macro metrics pretrain (2015-2018)
├── metric_train.csv         # Macro metrics training (2018-2023)
└── metric_test.csv          # Macro metrics test (2023-2026)
```

To preprocess raw data into separate CSVs:

```bash
uv run python script.py
```

## Training Pipeline

### Step 1: Train MacroHead (System 2)

```bash
uv run python train_macro_head.py
```

**Outputs**:

- `agent/save/macro_head.pt` — Model weights
- `agent/save/macro_scaler.pkl` — StandardScaler for macro features
- `runs/train_macro/` — Confusion matrices

### Step 2: Train the Agent (System 1)

```bash
uv run python train_agent.py
```

**Prerequisite**: `wandb` login required (for metric logging).

**Outputs**:

- `agent/save/agent_saved.pt` — Agent weights
- `agent/save/macro_head_1.pt` — MacroHead copy after fine-tuning

> **Note**: Training is forced to CPU despite GPU config. Batch size is 64, rollout is 2048 steps.

### Step 3: Backtest

```bash
uv run python test_agent.py
```

**Outputs**:

- `runs/test/backtest_result_{timestamp}.png` — Backtest chart

## Hyperparameter Search

### MacroHead (System 2)

```bash
uv run python train_macro_hyperparams.py
```

Uses Optuna (50 trials, 150 max trials) with time-series cross-validation (10 folds).

### Agent (System 1)

```bash
uv run python search_hyperparams_agent.py
```

Uses Optuna (50 trials, GPU if available). Search space:

- `lr`: learning rate (1e-6 to 1e-3)
- `gamma`: discount factor (0.80 to 0.99)
- `gae_lambda`: GAE lambda (0.80 to 0.99)
- `ent_coef`: entropy coefficient (0.01 to 0.9)
- `value_coef`: value loss coefficient (0.05 to 0.5)
- `belief_coef`: belief loss coefficient (0.01 to 0.5)
- `batch_size`: {64, 128, 256}

## Tests

```bash
uv run pytest tests/
```

> **Warning**: Tests require `data_off/unit_test/` which is not included. They will fail without this directory.

## Type Checking

```bash
mypy kairos/
```

> **Note**: `mypy.ini` references a nonexistent `rl_trade` directory. Use the command above directly.

## Docker

```bash
docker build -t kairos .
docker run --gpus all kairos
```

Uses the NVIDIA PyTorch container (`nvcr.io/nvidia/pytorch:25.08-py3`).

## Project Structure

```
Kairos/
├── agent/
│   ├── model.py              # MacroHead, Agent, FocalLoss
│   ├── ppo_belief.py         # PPO algorithm with auxiliary losses
│   ├── buffer.py             # Rollout buffer (pre-allocated tensors)
│   ├── save/                 # Saved weights (gitignored)
│   └── __init__.py
├── kairos/
│   ├── env.py                # Trading environment
│   ├── compute.py            # DSR reward, Sharpe ratio, max drawdown
│   ├── processing.py         # Unit conversions
│   └── __init__.py
├── tests/
│   └── test_env.py           # Unit tests (requires data)
├── data_off/
│   └── train_test/           # Datasets (gitignored)
├── runs/
│   └── test/                 # Backtest charts
├── train_macro_head.py       # System 2 training
├── train_agent.py            # System 1 training
├── test_agent.py             # Backtest
├── search_hyperparams_agent.py    # Optuna for Agent
├── train_macro_hyperparams.py     # Optuna for MacroHead
├── visualizer.py             # wandb visualizations (3D scatter)
├── script.py                 # Data preprocessing
├── pyproject.toml            # uv configuration
├── requirements.txt          # Full dependencies
├── dockerfile                # NVIDIA Docker image
└── AGENTS.md                 # Technical documentation
```

## Key Features

- **DSR-based reward**: Reward based on Deflated Sharpe Ratio with a 10-step warmup period
- **Action masking**: Ability to mask certain actions (liquidation protection)
- **FocalLoss**: Handles class imbalance for macro predictions
- **PPO clipping**: Training stabilization with configurable clip epsilon
- **Gradient clipping**: Max norm 1.0 to prevent gradient explosions
- **GAE**: Generalized Advantage Estimation for advantage computation
- **Learning rate decay**: Linear LR decay over the entire training duration

## Author

**Dar-rius** — mohamedtine17@gmail.com
