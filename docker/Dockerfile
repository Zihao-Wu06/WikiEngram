# CPU image (works anywhere; training is slow on CPU - use Dockerfile.cuda on an NVIDIA GPU).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf WIKIENGRAM_WORKSPACE=/data/workspace

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4" \
 && pip install ".[api,data,mcp]"

VOLUME ["/data"]
EXPOSE 8765 8000 8766
ENTRYPOINT ["wikiengram"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8765"]
