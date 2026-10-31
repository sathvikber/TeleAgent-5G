# ===========================================
# TeleAgent Docker Image
# ===========================================

# Reuse the stable training environment
FROM slm-lora:1.2

LABEL maintainer="Sapna Jha"
LABEL project="TeleAgent-5G"
LABEL version="0.1"

# Working directory
WORKDIR /workspace


COPY requirements.txt .

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]
# We will add TeleAgent-specific packages here later
#
# Example:
#
# RUN pip install fastapi uvicorn
# RUN pip install faiss-cpu
# RUN pip install chromadb
# RUN pip install prometheus-client
#
# For now, we keep the image identical to slm-lora:1.2
