# 1. Image OFFICIELLE pour Jetson Thor (ARM64) avec PyTorch et CUDA pré-compilés
FROM nvcr.io/nvidia/pytorch:24.09-py3-igpu

WORKDIR /kairos

# 2. Copier les dépendances et les données
COPY requirements.txt .
COPY data_off ./data_off

# 3. Installer les librairies (en forçant NumPy 1.x pour éviter le crash PyTorch)
RUN pip install -r requirements.txt
RUN pip install wandb "numpy<2"

# 4. Copier le reste du projet
COPY . .
