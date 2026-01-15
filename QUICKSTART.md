# Quick Start - Comece em 5 minutos! ⚡

## ⚠️ Erro ao executar a API?

Se você acabou de receber o erro:
```
ModuleNotFoundError: No module named 'langchain_anthropic'
```

**Solução rápida:**

### Opção 1: Script Automático (Mais Fácil)
```bash
# Execute o instalador
install.bat

# Aguarde finalizar (~2 min)
# Depois execute a API
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Opção 2: Manual
```bash
# 1. Atualizar pip
python -m pip install --upgrade pip

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Executar API
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🚀 Setup Inicial (Primeira Vez)

### Passo 1: Verificar Pré-requisitos

**Python 3.11+:**
```bash
python --version
# Deve mostrar: Python 3.11.x ou superior
```

**Ollama instalado (para embeddings locais):**
```bash
ollama --version
# Se não tiver: baixar em https://ollama.ai
```

### Passo 2: Instalar Dependências

**Automático:**
```bash
install.bat
```

**Ou manual:**
```bash
pip install -r requirements.txt
```

### Passo 3: Configurar API Keys

**Criar arquivo `.env`:**
```bash
# Copie o exemplo
copy .env.example .env

# Ou crie manualmente:
notepad .env
```

**Conteúdo do `.env`:**
```bash
# API Keys
OPENAI_API_KEY=sk-proj-sua-chave-aqui
ANTHROPIC_API_KEY=sk-ant-sua-chave-aqui

# Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434
```

### Passo 4: Baixar Modelo BGE-M3 (Ollama)

```bash
ollama pull bge-m3
```

Aguarde o download (~1.2GB) finalizar.

### Passo 5: Executar API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

**Saída esperada:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Application startup complete.
```

### Passo 6: Testar

**Abrir no navegador:**
- API Docs: http://localhost:8000/docs ✨
- Health: http://localhost:8000/health

**Ou via curl:**
```bash
curl http://localhost:8000/health
```

**Resposta esperada:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "available_contexts": 1,
  "embeddings_provider": "ollama"
}
```

---

## 🎯 Primeiro Uso - Fluxo Completo

### 1. Criar Contexto
```bash
curl -X POST http://localhost:8000/api/contexts \
  -H "Content-Type: application/json" \
  -d "{\"name\": \"meu_contexto\"}"
```

### 2. Indexar Documento

**Via Postman:**
1. Importar [postman_collection.json](postman_collection.json)
2. **Indexação** → **Indexar Documento - Contexto Específico**
3. Body → form-data:
   - `files`: [File] → Selecione PDF/DOCX
   - `context`: `meu_contexto`
   - `embedding_provider`: `ollama`
4. Send

**Via cURL (exemplo):**
```bash
curl -X POST http://localhost:8000/api/index \
  -F "files=@documento.pdf" \
  -F "context=meu_contexto" \
  -F "embedding_provider=ollama"
```

### 3. Fazer Pergunta

**Via Postman:**
1. **Consultas (Query)** → **Query Completa**
2. Body (JSON):
```json
{
  "question": "Qual o tema principal do documento?",
  "context": "meu_contexto",
  "llm_provider": "openai",
  "embedding_provider": "ollama",
  "top_k": 8
}
```
3. Send

**Via cURL:**
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Qual o tema principal?",
    "context": "meu_contexto",
    "llm_provider": "openai",
    "embedding_provider": "ollama"
  }'
```

### 4. Ver Resposta

```json
{
  "answer": "De acordo com o documento...",
  "sources": [
    {
      "content": "Trecho relevante do documento...",
      "file": "documento.pdf",
      "chunk": "1/50"
    }
  ],
  "context": "meu_contexto",
  "llm_provider": "openai",
  "embedding_provider": "ollama"
}
```

---

## 📱 Executar Interface Gradio (Alternativa)

Se preferir interface visual ao invés de API:

```bash
python app.py
```

Abre automaticamente: http://localhost:7860

---

## 🔧 Executar Ambos (API + Gradio)

**Terminal 1 - API:**
```bash
uvicorn src.api:app --port 8000 --reload
```

**Terminal 2 - Gradio:**
```bash
python app.py
```

**Acessar:**
- API: http://localhost:8000/docs
- Gradio: http://localhost:7860

---

## 🐳 Docker (Produção)

**Build e Run:**
```bash
docker-compose up --build
```

**Acessar:**
- Gradio: http://localhost:7860
- (Configure API no docker-compose.yml se necessário)

---

## 📚 Documentação

- **Como Funciona**: [COMO_FUNCIONA.md](COMO_FUNCIONA.md)
- **API REST**: [API_REST.md](API_REST.md)
- **Guia Postman**: [POSTMAN_GUIDE.md](POSTMAN_GUIDE.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **README**: [README.md](README.md)

---

## ⚡ Comandos Rápidos

### Verificar Status
```bash
curl http://localhost:8000/health
```

### Listar Contextos
```bash
curl http://localhost:8000/api/contexts
```

### Ver Swagger Docs
Abrir: http://localhost:8000/docs

### Parar API
Pressione `Ctrl+C` no terminal

### Logs Detalhados
```bash
uvicorn src.api:app --reload --log-level debug
```

---

## 🆘 Problemas Comuns

### ❌ ModuleNotFoundError
```bash
pip install -r requirements.txt
```

### ❌ Porta 8000 em uso
```bash
# Usar porta diferente
uvicorn src.api:app --port 8001 --reload
```

### ❌ Ollama não conecta
```bash
# Verificar se está rodando
curl http://localhost:11434

# Iniciar Ollama
ollama serve
```

### ❌ API Key inválida
```bash
# Editar .env com chave correta
notepad .env

# Reiniciar API (Ctrl+C e rodar novamente)
```

**Mais soluções:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## ✅ Checklist de Setup

- [ ] Python 3.11+ instalado
- [ ] Ollama instalado e rodando
- [ ] Modelo BGE-M3 baixado (`ollama pull bge-m3`)
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] Arquivo `.env` configurado
- [ ] API iniciada (`uvicorn src.api:app --reload`)
- [ ] Health check OK (`curl http://localhost:8000/health`)
- [ ] Postman collection importada (opcional)

---

## 🎉 Pronto!

Você agora tem:
- ✅ API REST rodando
- ✅ Sistema RAG funcional
- ✅ Embeddings locais (BGE-M3)
- ✅ Múltiplos contextos
- ✅ Documentação completa

**Próximos passos:**
1. Indexar seus documentos
2. Fazer perguntas
3. Integrar em sua aplicação

**Explore a documentação interativa:** http://localhost:8000/docs

---

## 📞 Ajuda

- Issues: https://github.com/seu-repo/issues
- Documentação: [COMO_FUNCIONA.md](COMO_FUNCIONA.md)
- Troubleshooting: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
