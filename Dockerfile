# RAG Simple - Dockerfile
FROM python:3.11-slim

# Metadados
LABEL maintainer="RAG Simple"
LABEL description="RAG System with LangChain, FAISS, Gradio and REST API"

# Variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Diretório de trabalho
WORKDIR /app

# Instala dependências do sistema (incluindo OCR com suporte a português)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpoppler-cpp-dev \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-por \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements primeiro (cache de camadas)
COPY requirements.txt .

# Instala dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia código da aplicação
COPY config.toml .
COPY app.py .
COPY src/ ./src/
COPY start.sh .

# Permissões do script de inicialização
RUN chmod +x start.sh

# Cria diretórios de dados com permissões corretas
RUN mkdir -p data/documents data/faiss_index data/temp_uploads && \
    chmod -R 777 data/

# Mantém root para evitar problemas de permissão com volumes montados
# Em produção, considere usar um usuário específico com UID/GID correspondente

# Expõe portas: Gradio (7860) + API REST (8000)
EXPOSE 7860 8000

# Health check - verifica ambos os serviços
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/ && curl -f http://localhost:8000/health || exit 1

# Comando de inicialização - executa ambos os serviços
CMD ["./start.sh"]
