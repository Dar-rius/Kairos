---
size_categories:
- 1M<n<10M
tags:
- finance
- crypto
- BTC
---
# Dataset Description

These datasets contain **financial data related to Bitcoin**. They are used to train a **reinforcement learning (RL) agent** whose objective is to learn how to trade the Bitcoin market.

The agent relies on an architecture inspired by the **System 1 / System 2 cognitive model**:

- **System 1**: responsible for fast decision-making during trading.
    
- **System 2**: a pre-trained neural network designed to analyze the market regime and provide macro-level context to improve the decisions made by System 1.
    
More specifically, **System 2** is trained to **identify the market regime for the next 24 hours** and provide this information to System 1 in order to guide its trading actions.

The data is organized into **two main groups of datasets**: **Macro-State** and **Micro-State**.

---
# Macro-State

The **Macro-State** group contains data describing the **macroeconomic evolution of the Bitcoin market as well as the state of the Bitcoin network**.

These datasets are used to **train System 2**, whose role is to detect market regimes.

The variables in these datasets are obtained from **transformations applied to raw on-chain and macro-financial data**, such as:

- **MVRV**
    
- **NVT**
    
- **Hash Rate**
    
- and other similar metrics.
    
The goal of these transformations is to **extract informative signals about the market structure and dynamics**, helping the agent better understand its environment.

Each row corresponds to **one day of market history**.

The datasets belonging to this group are:

- `metric_pretrain.csv`
    
- `metric_train.csv`
    
- `metric_test.csv`
    

---

# Micro-State

The **Micro-State** group contains data describing the **short-term microeconomic dynamics of the Bitcoin market**.

These datasets are used to **train System 1**, which performs the trading decisions.

Each row corresponds to **one hour of market history**.

The variables in this group are derived from **calculations based on OHLCV data (Open, High, Low, Close, Volume)**, allowing the extraction of indicators relevant for short-term trading decisions.

The datasets belonging to this group are:

- `price_train.csv`
    
- `price_test.csv`
    

---

# Dataset Time Range

The **overall market history** covered by the datasets spans from **2015/01/01 to 2026/03/03**, but the exact time ranges vary depending on the dataset:

1. **metric_pretrain.csv**: 2015/01/01 – 2018/07/19
    
2. **metric_train.csv**: 2018/07/20 – 2023/05/31
    
3. **metric_test.csv**: 2023/06/01 – 2026/03/03
    
4. **price_train.csv**: 2018/07/20 – 2023/05/31
    
5. **price_test.csv**: 2023/06/01 – 2026/03/03