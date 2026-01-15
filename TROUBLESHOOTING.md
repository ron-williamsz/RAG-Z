# Troubleshooting - Soluções Rápidas

## 🔴 Erro: ModuleNotFoundError: No module named 'langchain_anthropic'

### Causa
As dependências não foram instaladas ou há conflitos de versões.

### Solução

**Opção 1: Script de instalação automático (Recomendado)**
```bash
# Windows
install.bat

# Aguarde finalizar e siga as instruções
```

**Opção 2: Instalação manual**
```bash
# 1. Atualizar pip
python -m pip install --upgrade pip

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Se houver conflitos, force reinstalação
pip install --upgrade --force-reinstall -r requirements.txt
```

---

## 🔴 Erro: Conflitos de dependências (langchain-core, langsmith, etc.)

### Mensagem típica:
```
langchain 0.3.24 requires langchain-core<1.0.0,>=0.3.55, but you have langchain-core 1.2.1
```

### Solução

**Reinstalar com versões compatíveis:**
```bash
pip uninstall langchain langchain-core langchain-openai langchain-anthropic langchain-community langchain-ollama -y
pip install -r requirements.txt
```

---

## 🔴 Erro: Uvicorn não inicia / API não responde

### Causa
Porta 8000 já está em uso ou processo anterior não foi fechado.

### Solução

**Opção 1: Usar porta diferente**
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8001 --reload
```

**Opção 2: Matar processo na porta 8000**
```bash
# Windows PowerShell
$process = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($process) {
    Stop-Process -Id $process.OwningProcess -Force
}

# Depois execute normalmente
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

**Opção 3: Identificar e matar manualmente**
```bash
# Windows CMD
netstat -ano | findstr :8000
# Anote o PID (última coluna)
taskkill /PID [número_do_pid] /F
```

---

## 🔴 Erro 404: Contexto não encontrado

### Mensagem:
```json
{
  "detail": "Contexto 'cond_391' não encontrado"
}
```

### Solução

**1. Listar contextos disponíveis:**
```bash
curl http://localhost:8000/api/contexts
```

**2. Criar o contexto:**
```bash
curl -X POST http://localhost:8000/api/contexts \
  -H "Content-Type: application/json" \
  -d '{"name": "cond_391"}'
```

**3. Verificar se foi criado:**
```bash
curl http://localhost:8000/api/contexts
```

---

## 🔴 Erro 404: Índice FAISS não encontrado

### Mensagem:
```json
{
  "detail": "Índice FAISS não encontrado para contexto 'cond_391'"
}
```

### Causa
Contexto existe mas não tem documentos indexados.

### Solução

**Indexar documentos no contexto:**

**Via cURL:**
```bash
curl -X POST http://localhost:8000/api/index \
  -F "files=@documento.pdf" \
  -F "context=cond_391" \
  -F "embedding_provider=ollama"
```

**Via Postman:**
1. POST → `http://localhost:8000/api/index`
2. Body → form-data
3. files: [Select File] → escolha arquivo
4. context: `cond_391`
5. embedding_provider: `ollama`

---

## 🔴 Erro 500: OpenAI API Key não configurada

### Mensagem típica:
```
Error code: 401 - {'error': {'message': 'Incorrect API key provided'}}
```

### Solução

**1. Criar/editar arquivo `.env`:**
```bash
# .env
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434
```

**2. Verificar se API key está correta:**
- Acesse https://platform.openai.com/api-keys
- Crie nova chave se necessário
- Cole no `.env`

**3. Reiniciar servidor:**
```bash
# Ctrl+C para parar
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔴 Erro: Ollama não está rodando

### Mensagem:
```
Connection refused to http://localhost:11434
```

### Solução

**1. Verificar se Ollama está instalado:**
```bash
ollama --version
```

**2. Se não estiver, instalar:**
- Windows/Mac/Linux: https://ollama.ai/download

**3. Verificar se está rodando:**
```bash
curl http://localhost:11434
```

**4. Se não estiver, iniciar:**
```bash
# Windows - abrir Ollama Desktop App
# Ou executar no terminal:
ollama serve
```

**5. Baixar modelo BGE-M3:**
```bash
ollama pull bge-m3
```

**6. Verificar modelo:**
```bash
ollama list
# Deve aparecer: bge-m3
```

---

## 🔴 Erro: Python não encontrado

### Solução

**Verificar instalação:**
```bash
python --version
# ou
python3 --version
```

**Se não instalado:**
- Download: https://www.python.org/downloads/
- Durante instalação: ✅ **Add Python to PATH**

---

## 🔴 Erro: pip não reconhecido

### Solução

**Usar módulo pip:**
```bash
python -m pip install -r requirements.txt
```

**Ou adicionar pip ao PATH:**
```bash
# Windows - adicionar aos Scripts do Python
C:\Users\[usuario]\AppData\Local\Programs\Python\Python313\Scripts
```

---

## 🔴 Gradio não inicia

### Erro típico:
```
ModuleNotFoundError: No module named 'gradio'
```

### Solução

```bash
pip install gradio>=4.0.0
# Depois:
python app.py
```

---

## 🔴 Upload de arquivo falha (Postman/API)

### Causa
Campo `files` não está configurado como tipo **File** no Postman.

### Solução

**No Postman:**
1. Body → **form-data**
2. Key: `files`
3. **Mudar tipo de "Text" para "File"** no dropdown
4. Clicar em "Select Files"
5. Escolher arquivo(s)
6. Adicionar outros campos (context, embedding_provider) como Text

---

## 🔴 CORS Error (Frontend)

### Mensagem:
```
Access to fetch at 'http://localhost:8000' has been blocked by CORS policy
```

### Causa
Frontend está em origem diferente (ex: localhost:3000) e CORS não está configurado.

### Solução

**A API já tem CORS habilitado para todas as origens por padrão.**

Se ainda assim houver erro:

1. **Verificar se API está rodando:** http://localhost:8000/health

2. **Testar no navegador:**
```javascript
fetch('http://localhost:8000/health')
  .then(r => r.json())
  .then(console.log)
```

3. **Se necessário, restringir origens em produção** (editar `src/api.py`):
```python
allow_origins=["http://localhost:3000", "https://seu-frontend.com"]
```

---

## 🔴 Erro de Memória (MemoryError)

### Causa
Documento muito grande ou muitos chunks.

### Solução

**1. Aumentar chunk_size no `config.toml`:**
```toml
[chunking]
chunk_size = 1024  # Era 512
chunk_overlap = 100
```

**2. Processar documentos menores:**
- Dividir PDF grande em partes menores
- Usar `top_k` menor nas queries

**3. Limitar tamanho de arquivo no upload:**
No `src/api.py`, adicionar:
```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

if len(await file.read()) > MAX_FILE_SIZE:
    raise HTTPException(413, "Arquivo muito grande")
```

---

## 🔴 Respostas Lentas

### Causa
BGE-M3 (Ollama) pode ser lento em CPU.

### Soluções

**Opção 1: Usar OpenAI Embeddings (mais rápido, pago)**
```json
{
  "embedding_provider": "openai"
}
```

**Opção 2: Reduzir top_k**
```json
{
  "top_k": 3  # Em vez de 8
}
```

**Opção 3: GPU para Ollama**
- Verificar se Ollama está usando GPU
- Reinstalar Ollama com suporte CUDA/ROCm

---

## 🛠️ Comandos Úteis de Diagnóstico

### Verificar instalação de pacotes:
```bash
pip list | grep langchain
pip list | grep fastapi
pip list | grep gradio
```

### Verificar versões:
```bash
python --version
pip --version
ollama --version
```

### Testar imports:
```bash
python -c "import langchain; print('LangChain OK')"
python -c "import fastapi; print('FastAPI OK')"
python -c "import gradio; print('Gradio OK')"
python -c "from langchain_anthropic import ChatAnthropic; print('Anthropic OK')"
```

### Verificar API:
```bash
# Health check
curl http://localhost:8000/health

# Listar contextos
curl http://localhost:8000/api/contexts

# Docs
# Abrir no navegador: http://localhost:8000/docs
```

### Logs da API:
```bash
# Executar com logs detalhados
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

---

## 📞 Ainda com problemas?

### 1. Verificar logs do servidor
Ao executar `uvicorn`, leia os erros detalhados no console.

### 2. Verificar documentação
- [README.md](README.md)
- [API_REST.md](API_REST.md)
- [COMO_FUNCIONA.md](COMO_FUNCIONA.md)

### 3. Reinstalação completa
```bash
# Limpar tudo
pip uninstall -y -r requirements.txt

# Reinstalar
pip install -r requirements.txt

# Verificar
python -c "import src.api; print('API OK')"
```

### 4. Ambiente virtual limpo
```bash
# Criar novo venv
python -m venv venv_novo

# Ativar
# Windows:
venv_novo\Scripts\activate
# Linux/Mac:
source venv_novo/bin/activate

# Instalar
pip install -r requirements.txt

# Executar
uvicorn src.api:app --reload
```

---

## ✅ Checklist de Diagnóstico

Antes de reportar problema, verifique:

- [ ] Python >= 3.11 instalado
- [ ] pip atualizado (`python -m pip install --upgrade pip`)
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] Arquivo `.env` configurado com API keys
- [ ] Ollama rodando (se usar BGE-M3)
- [ ] Porta 8000 livre
- [ ] Contexto criado
- [ ] Documentos indexados
- [ ] Testar endpoint `/health` primeiro

---

## 📝 Template para Reportar Problema

```
**Ambiente:**
- OS: Windows 11 / macOS / Linux
- Python: 3.13
- Comando executado: uvicorn src.api:app --reload

**Erro:**
[Cole o erro completo aqui]

**O que já tentei:**
- [x] Reinstalar dependências
- [x] Verificar .env
- [ ] ...

**Logs:**
[Cole logs relevantes]
```

---

## 🎓 Recursos Adicionais

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Postman Collection**: [postman_collection.json](postman_collection.json)
- **Guia Postman**: [POSTMAN_GUIDE.md](POSTMAN_GUIDE.md)
