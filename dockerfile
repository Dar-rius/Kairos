FROM nvcr.io/nvidia/pytorch:25.08-py3

WORKDIR /build

WORKDIR /kairos
COPY requirements.txt .
RUN pip install --force-reinstall numpy==1.26.4 wandb && \
    pip install -r requirements.txt 

RUN pip uninstall numpy -y && \
    pip install numpy==1.26.4 --force-reinstall --no-binary :all:

COPY . .
