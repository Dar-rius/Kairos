FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.2-py3

WORKDIR /kairos

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . .

RUN uv pip install -r requirements.txt

CMD ["python3", "script.py"]
CMD ["python3", "train_macro_head.py"]
CMD ["python3", "train_agent.py"]
