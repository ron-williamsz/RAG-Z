# Guia Rápido - Testando API no Postman

## 📥 Como Importar

### 1. Importar Coleção

1. Abra o **Postman**
2. Clique em **Import** (canto superior esquerdo)
3. Arraste o arquivo `postman_collection.json` ou clique em **Upload Files**
4. Clique em **Import**

✅ A coleção "RAG API - Sistema de Consultas" será adicionada

### 2. Importar Environment (Opcional)

1. Clique em **Import** novamente
2. Arraste o arquivo `postman_environment.json`
3. Clique em **Import**
4. No canto superior direito, selecione **"RAG API - Local"** no dropdown de environments

---

## 🚀 Passo a Passo para Testar

### Pré-requisitos

Antes de testar, certifique-se que a API está rodando:

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Verifique que está rodando acessando: http://localhost:8000

---

## 📋 Ordem Recomendada de Testes

### 1️⃣ Verificar Status da API

**Pasta:** Health & Status → **Request:** Health Check

**Tipo:** GET

**URL:** `http://localhost:8000/health`

**Clique em:** Send

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

### 2️⃣ Listar Contextos Disponíveis

**Pasta:** Contextos → **Request:** Listar Todos os Contextos

**Tipo:** GET

**URL:** `http://localhost:8000/api/contexts`

**Resposta esperada:**
```json
{
  "contexts": [
    {
      "name": "default",
      "total_files": 0,
      "total_chunks": 0,
      "created_at": null,
      "last_updated": null
    }
  ],
  "total": 1
}
```

---

### 3️⃣ Criar Novo Contexto

**Pasta:** Contextos → **Request:** Criar Novo Contexto

**Tipo:** POST

**URL:** `http://localhost:8000/api/contexts`

**Body (JSON):**
```json
{
  "name": "cond_391"
}
```

**Como fazer:**
1. Selecione a aba **Body**
2. Escolha **raw**
3. Selecione **JSON** no dropdown
4. Cole o JSON acima
5. Clique em **Send**

**Resposta esperada:**
```json
{
  "status": "success",
  "message": "Contexto 'cond_391' criado com sucesso",
  "context_name": "cond_391"
}
```

---

### 4️⃣ Indexar Documentos

**Pasta:** Indexação → **Request:** Indexar Documento - Contexto Específico

**Tipo:** POST

**URL:** `http://localhost:8000/api/index`

**Como fazer:**

1. Selecione a aba **Body**
2. Escolha **form-data**
3. Configure os campos:

| Key | Type | Value |
|-----|------|-------|
| `files` | **File** ← importante! | Clique em "Select Files" e escolha um PDF/DOCX |
| `context` | Text | `cond_391` |
| `embedding_provider` | Text | `ollama` |

4. Clique em **Send**

⚠️ **IMPORTANTE:** Para o campo `files`, você DEVE mudar o tipo de "Text" para "**File**" no dropdown ao lado!

**Resposta esperada:**
```json
{
  "status": "success",
  "message": "Documentos indexados com sucesso no contexto 'cond_391'",
  "files_indexed": ["seu_arquivo.pdf"],
  "total_chunks": 234,
  "context": "cond_391",
  "embedding_provider": "ollama"
}
```

---

### 5️⃣ Fazer uma Consulta

**Pasta:** Consultas (Query) → **Request:** Query Completa - Todos os Parâmetros

**Tipo:** POST

**URL:** `http://localhost:8000/api/query`

**Body (JSON):**
```json
{
  "question": "Posso ter cachorro no condomínio?",
  "context": "cond_391",
  "llm_provider": "openai",
  "embedding_provider": "ollama",
  "top_k": 8,
  "return_sources": true
}
```

**Como fazer:**
1. Aba **Body** → **raw** → **JSON**
2. Cole o JSON acima
3. **Ajuste a pergunta** para algo relevante aos seus documentos
4. Clique em **Send**

**Resposta esperada:**
```json
{
  "answer": "De acordo com o Regimento Interno do condomínio...",
  "sources": [
    {
      "content": "Artigo 15 - Não é permitido ter animais...",
      "file": "regimento_interno.pdf",
      "chunk": "42/150"
    }
  ],
  "context": "cond_391",
  "llm_provider": "openai",
  "embedding_provider": "ollama",
  "context_format": "toon"
}
```

---

### 6️⃣ Ver Estatísticas do Contexto

**Pasta:** Contextos → **Request:** Estatísticas de Contexto Específico

**Tipo:** GET

**URL:** `http://localhost:8000/api/contexts/cond_391/stats`

**Como fazer:**
1. Substitua `cond_391` na URL pelo seu contexto
2. Clique em **Send**

**Resposta esperada:**
```json
{
  "name": "cond_391",
  "total_files": 1,
  "total_chunks": 234,
  "files": [...],
  "created_at": "2024-01-15T10:30:00",
  "last_updated": "2024-01-20T15:45:00"
}
```

---

## 🎯 Testes Recomendados por Cenário

### Cenário 1: Teste Rápido (5 min)

1. ✅ Health Check
2. ✅ Listar Contextos
3. ✅ Query Básica - Default

### Cenário 2: Teste Completo (15 min)

1. ✅ Health Check
2. ✅ Criar Novo Contexto
3. ✅ Indexar Documento
4. ✅ Query Completa
5. ✅ Ver Estatísticas
6. ✅ Listar Contextos

### Cenário 3: Teste de Erros (10 min)

Use a pasta **"Exemplos de Erros"** para testar:
- ❌ Erro 400 - Pergunta Vazia
- ❌ Erro 400 - Provider Inválido
- ❌ Erro 404 - Contexto Inexistente
- ❌ Erro 409 - Contexto Já Existe
- ❌ Erro 403 - Deletar Contexto Default

---

## 🔄 Variações de Teste

### Testar Diferentes LLMs

**OpenAI GPT-4o:**
```json
{
  "question": "Sua pergunta",
  "llm_provider": "openai"
}
```

**Anthropic Claude:**
```json
{
  "question": "Sua pergunta",
  "llm_provider": "anthropic"
}
```

### Testar Diferentes Embeddings

**Ollama BGE-M3 (Local, Grátis):**
```json
{
  "question": "Sua pergunta",
  "embedding_provider": "ollama"
}
```

**OpenAI Embeddings (Pago):**
```json
{
  "question": "Sua pergunta",
  "embedding_provider": "openai"
}
```

### Ajustar Número de Chunks Recuperados

**Poucos chunks (mais rápido):**
```json
{
  "question": "Pergunta específica",
  "top_k": 3
}
```

**Muitos chunks (mais contexto):**
```json
{
  "question": "Pergunta abrangente",
  "top_k": 15
}
```

---

## 🛠️ Troubleshooting

### ❌ Erro: "Could not get any response"

**Causa:** API não está rodando

**Solução:**
```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### ❌ Erro 404: "Contexto não encontrado"

**Causa:** Contexto não existe

**Solução:**
1. Execute: Listar Todos os Contextos
2. Use um contexto da lista ou crie um novo

### ❌ Erro 404: "Índice FAISS não encontrado"

**Causa:** Contexto existe mas não tem documentos indexados

**Solução:**
1. Indexe documentos usando o endpoint `/api/index`

### ❌ Erro 500: "Erro ao processar query"

**Causas possíveis:**
- API key do OpenAI/Anthropic não configurada
- Ollama não está rodando (para BGE-M3)
- Documento corrompido

**Solução:**
1. Verifique arquivo `.env` com as API keys
2. Verifique se Ollama está rodando: `http://localhost:11434`
3. Veja os logs do servidor para detalhes

---

## 📊 Dicas de Uso

### 1. Salvar Respostas

Após receber uma resposta, você pode salvá-la clicando em **Save Response** → **Save as Example**

### 2. Organizar Testes

Crie uma **pasta personalizada** na coleção para seus testes:
- Clique com botão direito na coleção
- **Add Folder**
- Nomeie (ex: "Meus Testes")
- Duplique requests e modifique

### 3. Usar Variáveis

No Body ou URL, use variáveis do environment:

```json
{
  "question": "Teste",
  "context": "{{context_cond}}",
  "llm_provider": "{{llm_provider}}"
}
```

Para editar variáveis:
1. Canto superior direito → **Environments**
2. Selecione "RAG API - Local"
3. Edite os valores

### 4. Visualizar Responses

Postman formata JSON automaticamente. Use as abas:
- **Pretty** - Formatado e colorido
- **Raw** - Texto puro
- **Preview** - Renderizado

---

## 📝 Exemplos de Perguntas para Testar

### Para Condomínio:
- "Posso ter animais de estimação?"
- "Qual o horário da piscina?"
- "Quais são as regras de barulho?"
- "Como funciona o estacionamento de visitantes?"
- "Quais documentos preciso para fazer uma reforma?"

### Para Empresa:
- "Qual a política de férias?"
- "Como solicitar reembolso?"
- "Quais são os benefícios oferecidos?"
- "Qual o horário de trabalho?"

### Testes Genéricos:
- "Resuma os principais pontos deste documento"
- "Quais são as regras mais importantes?"
- "O que diz sobre [tema específico]?"

---

## 🎓 Recursos Adicionais

### Documentação Completa da API
Consulte [API_REST.md](API_REST.md) para documentação detalhada.

### Swagger UI (Alternativa ao Postman)
Acesse http://localhost:8000/docs para testar direto no navegador.

### ReDoc
Acesse http://localhost:8000/redoc para ver documentação interativa.

---

## ✅ Checklist de Teste

Use este checklist para garantir que testou tudo:

- [ ] Health check funcionando
- [ ] Listar contextos (default aparece)
- [ ] Criar novo contexto
- [ ] Indexar documento em contexto
- [ ] Fazer query simples
- [ ] Fazer query com todos os parâmetros
- [ ] Ver estatísticas de contexto
- [ ] Testar com OpenAI
- [ ] Testar com Claude (se tiver API key)
- [ ] Testar diferentes valores de top_k
- [ ] Testar sem retornar fontes
- [ ] Ver exemplos de erros
- [ ] Deletar contexto (não default)

---

## 🚀 Pronto para Usar!

Você agora tem uma coleção completa do Postman com:

✅ **26+ requests** prontos para uso
✅ **Exemplos de sucesso** e de erros
✅ **Documentação** inline em cada request
✅ **Environment** configurável
✅ **Organizados por categoria**

**Bons testes! 🎉**
