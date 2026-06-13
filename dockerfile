FROM nvcr.io/nvidia/pytorch:25.08-py3

WORKDIR /kairos
COPY requirements.txt .
RUN pip install --force-reinstall numpy==1.26.4 wandb && \
    pip install -r requirements.txt 

RUN pip uninstall numpy -y && \
    pip install numpy==1.26.4 --force-reinstall --no-binary :all:

COPY . .

CMD ["/bin/bash", "-c", "if [ -f /run/secrets/wandb.key ]; then export WANDB_API_KEY=$(cat /run/secrets/wandb.key | xargs); fi && exec /bin/bash"]
