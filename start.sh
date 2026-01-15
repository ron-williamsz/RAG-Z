#!/bin/bash
# Script de inicialização - Executa Gradio + API REST simultaneamente

echo "============================================"
echo "  RAG System - Iniciando Servicos"
echo "============================================"
echo ""

# Diretório da aplicação
cd /app

# Função para tratar SIGTERM
cleanup() {
    echo ""
    echo "Encerrando servicos..."
    kill $GRADIO_PID $API_PID 2>/dev/null
    wait $GRADIO_PID $API_PID 2>/dev/null
    echo "Servicos encerrados."
    exit 0
}

# Captura sinais de encerramento
trap cleanup SIGTERM SIGINT

echo "[1/2] Iniciando Gradio (porta 7860)..."
python app.py &
GRADIO_PID=$!

# Aguarda um pouco para o Gradio iniciar
sleep 3

echo "[2/2] Iniciando API REST (porta 8000)..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

echo ""
echo "============================================"
echo "  Servicos Iniciados com Sucesso!"
echo "============================================"
echo ""
echo "  Gradio UI:    http://localhost:7860"
echo "  API REST:     http://localhost:8000"
echo "  API Docs:     http://localhost:8000/docs"
echo "  Health Check: http://localhost:8000/health"
echo ""
echo "============================================"
echo ""

# Aguarda os processos
wait $GRADIO_PID $API_PID
