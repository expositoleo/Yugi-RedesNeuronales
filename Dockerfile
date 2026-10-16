FROM python:3.10-slim

# Evitar prompts interactivos y buffering en los logs de la consola
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Dependencias mínimas del sistema para OpenCV (headless) y Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar PyTorch y torchvision versión CPU primero
# Esto ahorra más de 2 GB de descarga de CUDA innecesaria y acelera el despliegue
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copiar requirements e instalar el resto de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del proyecto, frontend, backend y artefactos
COPY backend/ ./backend/
COPY dev/ ./dev/
COPY frontend/ ./frontend/
COPY artifacts/ ./artifacts/
COPY data/ ./data/

# El puerto 8000 (o el provisto por la variable $PORT del servicio cloud)
EXPOSE 8000

# Arranque del servidor web FastAPI
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
