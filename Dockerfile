FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY main.py ./main.py
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "main.py"]
