@echo off
echo ================================
echo  RAG API - Instalacao Automatica
echo ================================
echo.

echo [1/4] Atualizando pip...
python -m pip install --upgrade pip

echo.
echo [2/4] Instalando dependencias do requirements.txt...
pip install -r requirements.txt

echo.
echo [3/4] Verificando instalacao...
python -c "import langchain; import fastapi; import gradio; print('OK - Todos os pacotes principais instalados!')"

if %errorlevel% neq 0 (
    echo.
    echo ERRO: Algumas dependencias nao foram instaladas corretamente.
    echo Por favor, execute manualmente: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo [4/4] Verificando arquivo .env...
if not exist .env (
    echo AVISO: Arquivo .env nao encontrado!
    echo.
    echo Criando .env de exemplo...
    echo # API Keys > .env
    echo OPENAI_API_KEY=sua-chave-aqui >> .env
    echo ANTHROPIC_API_KEY=sua-chave-aqui >> .env
    echo. >> .env
    echo # Ollama >> .env
    echo OLLAMA_BASE_URL=http://localhost:11434 >> .env
    echo.
    echo Arquivo .env criado! Por favor, edite e adicione suas API keys.
) else (
    echo OK - Arquivo .env encontrado!
)

echo.
echo ================================
echo  Instalacao Concluida!
echo ================================
echo.
echo Proximo passo: Execute a API com:
echo   uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
echo.
echo Ou execute a interface Gradio:
echo   python app.py
echo.
pause
