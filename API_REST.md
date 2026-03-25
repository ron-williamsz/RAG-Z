# API REST - Sistema RAG

## Índice
1. [Visão Geral](#visão-geral)
2. [Instalação e Configuração](#instalação-e-configuração)
3. [Como Executar](#como-executar)
4. [Endpoints Disponíveis](#endpoints-disponíveis)
   - [Root](#1-root)
   - [Health Check](#2-health-check)
   - [Consultar (Query)](#3-consultar-query)
   - [Consulta Hierárquica](#4-consulta-hierárquica)
   - [Indexar Documentos](#5-indexar-documentos)
   - [Listar Contextos](#6-listar-contextos)
   - [Criar Contexto](#7-criar-contexto)
   - [Deletar Contexto](#8-deletar-contexto)
   - [Estatísticas de Contexto](#9-estatísticas-de-contexto)
   - [Verificação - Extrair Referência](#10-verificação---extrair-referência)
   - [Verificação - Comparar Target](#11-verificação---comparar-target)
   - [Verificação - Sessões Ativas](#12-verificação---sessões-ativas)
   - [Admin - Prompt Anônimo](#13-admin---prompt-anônimo)
5. [Exemplos de Uso](#exemplos-de-uso)
6. [Integração Docker](#integração-docker)
7. [Tratamento de Erros](#tratamento-de-erros)
8. [Segurança](#segurança)

---

## Visão Geral

A API REST permite integrar o sistema RAG com outras aplicações através de endpoints HTTP. Desenvolvida com **FastAPI**, oferece:

- ✅ **Consultas RAG** - Faça perguntas e receba respostas contextualizadas
- ✅ **Consulta Hierárquica** - Busca em cascata com hierarquia legal (Código Civil → Lei de Condomínios → Documentos do Condomínio)
- ✅ **Indexação de documentos** - Upload e indexação via API com suporte a hierarquia legal
- ✅ **Gerenciamento de contextos** - Crie, liste e delete contextos
- ✅ **Verificação de documentos** - Motor de verificação com extração de referência e comparação
- ✅ **Governança e perfis** - Controle de acesso por perfil (anônimo, autenticado, admin)
- ✅ **Prompt anônimo customizável** - Administração do prompt para usuários anônimos
- ✅ **Múltiplos provedores** - Escolha LLM e embeddings provider
- ✅ **Histórico de conversa** - Suporte a contexto conversacional
- ✅ **Documentação automática** - Swagger UI e ReDoc
- ✅ **CORS habilitado** - Acesso de qualquer origem
- ✅ **Validação automática** - Pydantic models

**Base URL**: `http://localhost:8000`

**Documentação interativa**:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Instalação e Configuração

### 1. Dependências

As dependências já estão no [requirements.txt](requirements.txt):

```bash
pip install -r requirements.txt
```

Principais pacotes adicionados:
- `fastapi>=0.115.0` - Framework web
- `uvicorn[standard]>=0.30.0` - Servidor ASGI
- `python-multipart>=0.0.9` - Upload de arquivos

### 2. Variáveis de Ambiente

Configure no arquivo `.env`:

```bash
# LLMs
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. Configuração (config.toml)

Certifique-se que [config.toml](config.toml) está configurado corretamente:

```toml
[embeddings]
provider = "ollama"
model = "bge-m3"
base_url = "http://host.docker.internal:11434"

[llm.openai]
model = "gpt-4o"
temperature = 0.3
max_tokens = 4096

[retrieval]
top_k = 8
score_threshold = 0.7
```

---

## Como Executar

### Opção 1: Local (Desenvolvimento)

```bash
# Windows PowerShell
cd g:\.shortcut-targets-by-id\18aotPEyvREJzH6leCNgftNtNuRdU5hY9\Documents\robos\RAG-new

# Executar com reload automático
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

**Saída esperada:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Acesse:
- API: http://localhost:8000
- Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Opção 2: Docker (Produção)

Adicione ao [docker-compose.yml](docker-compose.yml):

```yaml
services:
  rag-api:
    build: .
    container_name: rag-api
    ports:
      - "8000:8000"
    volumes:
      - ./data/faiss_index:/app/data/faiss_index
      - ./data/documents:/app/data/documents
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    command: uvicorn src.api:app --host 0.0.0.0 --port 8000
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

**Executar:**
```bash
docker-compose up rag-api
```

### Opção 3: Rodar API + Gradio Juntos

Você pode rodar ambos simultaneamente:

```bash
# Terminal 1 - API
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Gradio
python app.py
```

- **API**: http://localhost:8000
- **Gradio**: http://localhost:7860

---

## Endpoints Disponíveis

### 1. Root

**GET** `/`

Informações básicas da API.

**Response:**
```json
{
  "name": "RAG API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

### 2. Health Check

**GET** `/health`

Verifica se a API está funcionando.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "available_contexts": 3,
  "embeddings_provider": "ollama"
}
```

**Códigos de Status:**
- `200` - Saudável
- `503` - Serviço indisponível

---

### 3. Consultar (Query)

**POST** `/api/query`

Executa uma consulta simples no sistema RAG (sem hierarquia legal).

**Request Body:**
```json
{
  "question": "Posso ter cachorro no condomínio?",
  "context": "cond_391",
  "llm_provider": "openai",
  "embedding_provider": "ollama",
  "top_k": 8,
  "return_sources": true,
  "use_legal_hierarchy": false,
  "show_source": false,
  "fluent_mode": true,
  "conversation_history": [
    {"role": "user", "content": "Olá"},
    {"role": "assistant", "content": "Olá! Como posso ajudar?"}
  ],
  "conversation_id": "conv_abc123",
  "user_id": "user_001",
  "is_authenticated": true,
  "is_admin": false
}
```

**Parâmetros:**
| Campo | Tipo | Obrigatório | Padrão | Descrição |
|-------|------|------------|--------|-----------|
| `question` | string | ✅ Sim | - | Pergunta do usuário |
| `context` | string | ❌ Não | "default" | Nome do contexto |
| `llm_provider` | string | ❌ Não | "openai" | LLM (openai, anthropic) |
| `embedding_provider` | string | ❌ Não | "ollama" | Embeddings (ollama, openai) |
| `top_k` | integer | ❌ Não | 8 | Chunks a recuperar (1-20) |
| `return_sources` | boolean | ❌ Não | true | Retornar fontes no JSON |
| `use_legal_hierarchy` | boolean | ❌ Não | false | Usar hierarquia legal |
| `show_source` | boolean | ❌ Não | false | Mencionar fonte na resposta textual |
| `fluent_mode` | boolean | ❌ Não | true | Resposta fluida sem jargões técnicos |
| `conversation_history` | array | ❌ Não | null | Histórico de mensagens `[{role, content}]` |
| `conversation_id` | string | ❌ Não | null | ID da conversa no chatbot externo |
| `user_id` | string | ❌ Não | null | ID do usuário |
| `is_authenticated` | boolean | ❌ Não | false | Se o usuário está autenticado |
| `is_admin` | boolean | ❌ Não | false | Se o usuário é administrador |

**Response:**
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

**Códigos de Status:**
- `200` - Sucesso
- `400` - Requisição inválida (pergunta vazia, provider inválido)
- `404` - Contexto não encontrado ou sem índice
- `500` - Erro interno

---

### 4. Consulta Hierárquica

**POST** `/api/query/hierarchical`

Executa consulta com busca em cascata respeitando a hierarquia legal. É o endpoint principal para integrações com chatbots.

**Hierarquia de busca:**
1. **Contexto do Condomínio** (convenção, regimento, atas)
2. **Lei de Condomínios** (Lei 4.591/64)
3. **Código Civil** (Lei 10.406/2002)

A busca é feita em cascata: busca primeiro no contexto principal e, se não encontrar resultados com score suficiente (>= 0.7), busca nos níveis seguintes da hierarquia.

**Request Body:**
```json
{
  "question": "Posso ter cachorro no condomínio?",
  "context": "cond_0388",
  "llm_provider": "openai",
  "embedding_provider": "ollama",
  "top_k_per_context": 3,
  "return_sources": true,
  "show_source": true,
  "fluent_mode": true,
  "hierarchy_level": null,
  "strict_hierarchy": false,
  "conversation_history": [
    {"role": "user", "content": "Quais as regras do condomínio?"},
    {"role": "assistant", "content": "O condomínio possui regras sobre..."}
  ],
  "conversation_id": "conv_abc123",
  "user_id": "user_001",
  "is_authenticated": true,
  "is_admin": false
}
```

**Parâmetros:**
| Campo | Tipo | Obrigatório | Padrão | Descrição |
|-------|------|------------|--------|-----------|
| `question` | string | ✅ Sim | - | Pergunta do usuário |
| `context` | string | ❌ Não | "zangari_website" | Contexto principal (ex: cond_0388, zangari_website) |
| `llm_provider` | string | ❌ Não | "openai" | LLM (openai, anthropic) |
| `embedding_provider` | string | ❌ Não | "ollama" | Embeddings (ollama, openai) |
| `top_k_per_context` | integer | ❌ Não | 3 | Chunks por contexto hierárquico (1-10) |
| `return_sources` | boolean | ❌ Não | true | Retornar fontes no JSON |
| `show_source` | boolean | ❌ Não | false | Mencionar fonte na resposta textual (ex: "De acordo com a convenção...") |
| `fluent_mode` | boolean | ❌ Não | true | Resposta fluida sem hierarquia técnica |
| `hierarchy_level` | string | ❌ Não | null | Nível específico: convencao, regimento_interno, codigo_civil, lei_condominios, ata_assembleia, avisos |
| `strict_hierarchy` | boolean | ❌ Não | false | Se true, retorna apenas do nível solicitado |
| `conversation_history` | array | ❌ Não | null | Histórico de mensagens `[{role, content}]` |
| `conversation_id` | string | ❌ Não | null | ID da conversa no chatbot externo |
| `user_id` | string | ❌ Não | null | ID do usuário |
| `is_authenticated` | boolean | ❌ Não | false | Se o usuário está autenticado |
| `is_admin` | boolean | ❌ Não | false | Se o usuário é administrador |

**Response:**
```json
{
  "answer": "De acordo com a Convenção do Condomínio, animais de estimação são permitidos...",
  "sources": [
    {
      "content": "Artigo 15 - É permitida a manutenção de animais...",
      "file": "convencao.pdf",
      "chunk": "12/50"
    }
  ],
  "contexts_searched": ["cond_0388", "lei_condominios", "codigo_civil"],
  "contexts_with_results": ["cond_0388"],
  "hierarchy_applied": true,
  "hierarchy_level": null,
  "found_in_requested_level": null,
  "fallback_used": null,
  "user_profile": "authenticated",
  "llm_provider": "openai",
  "embedding_provider": "ollama",
  "context_format": "hierarchical",
  "conversation_id": "conv_abc123"
}
```

**Perfis de Acesso (`user_profile`):**
| Perfil | Descrição |
|--------|-----------|
| `anonymous` | Usuário não autenticado - acesso apenas a contextos públicos |
| `authenticated` | Usuário autenticado - acesso ao condomínio associado |
| `admin` | Administrador - acesso a todos os contextos |

**Códigos de Status:**
- `200` - Sucesso
- `400` - Requisição inválida (pergunta vazia, provider inválido)
- `403` - Acesso negado ao contexto para o perfil do usuário
- `500` - Erro interno

---

### 5. Indexar Documentos

**POST** `/api/index`

Faz upload e indexa documentos em um contexto. Suporta metadados de hierarquia legal.

**Content-Type:** `multipart/form-data`

**Form Data:**
| Campo | Tipo | Obrigatório | Padrão | Descrição |
|-------|------|------------|--------|-----------|
| `files` | file(s) | ✅ Sim | - | Arquivos para indexar (PDF, DOCX, XLSX, TXT, MD) |
| `context` | string | ❌ Não | "default" | Nome do contexto |
| `embedding_provider` | string | ❌ Não | "ollama" | Provedor de embeddings (ollama, openai) |
| `hierarchy_level` | string | ❌ Não | null | Nível hierárquico: convencao, regimento_interno, ata_assembleia, avisos |

**Exemplo com curl:**
```bash
curl -X POST http://localhost:8000/api/index \
  -F "files=@regimento.pdf" \
  -F "files=@ata_assembleia.docx" \
  -F "context=cond_391" \
  -F "embedding_provider=ollama" \
  -F "hierarchy_level=regimento_interno"
```

**Response:**
```json
{
  "status": "success",
  "message": "Documentos indexados com sucesso no contexto 'cond_391'",
  "files_indexed": ["regimento.pdf", "ata_assembleia.docx"],
  "total_chunks": 234,
  "context": "cond_391",
  "embedding_provider": "ollama"
}
```

**Códigos de Status:**
- `200` - Sucesso
- `400` - Nenhum arquivo ou provider inválido
- `500` - Erro ao indexar

---

### 6. Listar Contextos

**GET** `/api/contexts`
**Tags:** `Contexts`

Lista todos os contextos disponíveis com estatísticas.

**Response:**
```json
{
  "contexts": [
    {
      "name": "default",
      "total_files": 5,
      "total_chunks": 123,
      "created_at": "2024-01-15T10:30:00",
      "last_updated": "2024-01-20T15:45:00"
    },
    {
      "name": "cond_391",
      "total_files": 12,
      "total_chunks": 456,
      "created_at": "2024-01-18T14:20:00",
      "last_updated": "2024-01-22T09:15:00"
    }
  ],
  "total": 2
}
```

**Códigos de Status:**
- `200` - Sucesso
- `500` - Erro interno

---

### 7. Criar Contexto

**POST** `/api/contexts`
**Tags:** `Contexts`

Cria um novo contexto vazio.

**Request Body:**
```json
{
  "name": "cond_392"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Contexto 'cond_392' criado com sucesso",
  "context_name": "cond_392"
}
```

**Códigos de Status:**
- `200` - Sucesso
- `409` - Contexto já existe
- `500` - Erro interno

---

### 8. Deletar Contexto

**DELETE** `/api/contexts/{context_name}`
**Tags:** `Contexts`

Deleta um contexto e todos os seus dados.

⚠️ **ATENÇÃO**: Operação irreversível!

**Exemplo:**
```bash
curl -X DELETE http://localhost:8000/api/contexts/cond_392
```

**Response:**
```json
{
  "status": "success",
  "message": "Contexto 'cond_392' deletado com sucesso"
}
```

**Restrições:**
- ❌ Não pode deletar o contexto "default"

**Códigos de Status:**
- `200` - Sucesso
- `403` - Tentativa de deletar "default"
- `404` - Contexto não encontrado
- `500` - Erro interno

---

### 9. Estatísticas de Contexto

**GET** `/api/contexts/{context_name}/stats`
**Tags:** `Contexts`

Obtém estatísticas detalhadas de um contexto.

**Exemplo:**
```bash
curl http://localhost:8000/api/contexts/cond_391/stats
```

**Response:**
```json
{
  "name": "cond_391",
  "total_files": 12,
  "total_chunks": 456,
  "files": [
    {
      "name": "regimento.pdf",
      "chunks": 45,
      "indexed_at": "2024-01-15T10:35:00"
    },
    {
      "name": "ata_2024.pdf",
      "chunks": 23,
      "indexed_at": "2024-01-20T15:45:00"
    }
  ],
  "created_at": "2024-01-15T10:30:00",
  "last_updated": "2024-01-20T15:45:00",
  "embeddings_model": "bge-m3",
  "embeddings_provider": "ollama"
}
```

**Códigos de Status:**
- `200` - Sucesso
- `404` - Contexto não encontrado
- `500` - Erro interno

---

### 10. Verificação - Extrair Referência

**POST** `/api/verify/extract-reference`
**Tags:** `Verification`

Etapa 1 do motor de verificação. Extrai entidades de referência de um documento base para posterior comparação.

**Content-Type:** `multipart/form-data`

**Form Data:**
| Campo | Tipo | Obrigatório | Padrão | Descrição |
|-------|------|------------|--------|-----------|
| `file` | file | ✅ Sim | - | Documento base de referência (PDF, DOCX, etc.) |
| `request_data` | string (JSON) | ✅ Sim | - | JSON com parâmetros de extração |

**Formato do `request_data` (JSON):**
```json
{
  "extraction_query": "lista de nomes de funcionários",
  "llm_provider": "openai",
  "session_ttl": 3600
}
```

| Campo | Tipo | Obrigatório | Padrão | Descrição |
|-------|------|------------|--------|-----------|
| `extraction_query` | string | ✅ Sim | - | Query em linguagem natural (ex: "lista de nomes") |
| `llm_provider` | string | ❌ Não | "openai" | LLM (openai, anthropic) |
| `session_ttl` | integer | ❌ Não | 3600 | Tempo de vida da sessão em segundos |

**Response:**
```json
{
  "session_id": "sess_abc123",
  "entity_type": "employee_names",
  "entities": ["João Silva", "Maria Santos", "Pedro Oliveira"],
  "total_entities": 3,
  "base_document": "lista_funcionarios.pdf",
  "expires_at": "2024-01-15T11:30:00",
  "message": "Successfully extracted 3 employee_names"
}
```

**Códigos de Status:**
- `200` - Sucesso
- `400` - JSON inválido ou dados de requisição inválidos
- `500` - Erro na extração

---

### 11. Verificação - Comparar Target

**POST** `/api/verify/compare-target`
**Tags:** `Verification`

Etapa 2 do motor de verificação. Compara documentos alvo contra a sessão de referência criada na etapa 1.

**Content-Type:** `multipart/form-data`

**Form Data:**
| Campo | Tipo | Obrigatório | Padrão | Descrição |
|-------|------|------------|--------|-----------|
| `files` | file(s) | ✅ Sim | - | Documentos alvo para verificar |
| `session_id` | string | ✅ Sim | - | Session ID da etapa extract-reference |
| `llm_provider` | string | ❌ Não | "openai" | LLM (openai, anthropic) |
| `strictness` | float | ❌ Não | 0.7 | Rigidez do match (0.5=tolerante, 1.0=estrito) |

**Response:**
```json
{
  "session_id": "sess_abc123",
  "results": [
    {
      "target_document": "folha_pagamento.pdf",
      "status": "verified",
      "overall_confidence": 0.95,
      "matches": [...],
      "mismatches": [...]
    }
  ],
  "summary_statistics": {
    "total_targets": 1,
    "verified": 1,
    "partial_match": 0,
    "mismatch": 0,
    "average_confidence": 0.95
  },
  "message": "Compared 1 target document(s)"
}
```

**Códigos de Status:**
- `200` - Sucesso
- `400` - Sem arquivos, strictness inválido, ou provider inválido
- `404` - Sessão não encontrada
- `500` - Erro na comparação

---

### 12. Verificação - Sessões Ativas

**GET** `/api/verify/sessions`
**Tags:** `Verification`

Retorna informações sobre sessões de verificação ativas.

**Response:**
```json
{
  "active_sessions": 2,
  "session_ids": ["sess_abc123", "sess_def456"]
}
```

**Códigos de Status:**
- `200` - Sucesso

---

### 13. Admin - Prompt Anônimo

Endpoints para gerenciar o prompt de sistema usado para usuários anônimos.

#### 13.1 Obter Prompt Anônimo

**GET** `/api/admin/rag/anonymous-prompt`
**Tags:** `Admin`

Retorna o prompt anônimo atual, o padrão e um preview.

**Response:**
```json
{
  "current_prompt": "",
  "default_prompt": "Você é o assistente virtual público do Grupo Zangari...",
  "using_default": true,
  "available_variables": ["{context_name}", "{context_label}", "{available_contexts}"],
  "preview": "Você é o assistente virtual público do Grupo Zangari.\n\nCONTEXTO ATIVO: Website Zangari (zangari_website)..."
}
```

#### 13.2 Atualizar Prompt Anônimo

**PUT** `/api/admin/rag/anonymous-prompt`
**Tags:** `Admin`

Define um prompt customizado para usuários anônimos.

**Request Body:**
```json
{
  "prompt": "Você é o assistente do Grupo Zangari.\nCONTEXTO: {context_label} ({context_name})\nCONTEXTOS: {available_contexts}\nResponda apenas com base nos documentos."
}
```

**Variáveis disponíveis no template:**
| Variável | Descrição |
|----------|-----------|
| `{context_name}` | Nome técnico do contexto (ex: lei_condominios) |
| `{context_label}` | Nome amigável (ex: Lei de Condomínios) |
| `{available_contexts}` | Lista formatada dos contextos públicos disponíveis |

**Response:**
```json
{
  "status": "success",
  "message": "Prompt anônimo atualizado com sucesso",
  "preview": "Você é o assistente do Grupo Zangari..."
}
```

**Códigos de Status:**
- `200` - Sucesso
- `400` - Variável inválida no template

#### 13.3 Resetar Prompt Anônimo

**DELETE** `/api/admin/rag/anonymous-prompt`
**Tags:** `Admin`

Reseta o prompt anônimo para o padrão do sistema.

**Response:**
```json
{
  "status": "success",
  "message": "Prompt anônimo resetado para o padrão",
  "default_prompt": "Você é o assistente virtual público do Grupo Zangari..."
}
```

---

## Exemplos de Uso

### Python (requests)

```python
import requests

# 1. Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# 2. Criar contexto
response = requests.post(
    "http://localhost:8000/api/contexts",
    json={"name": "meu_contexto"}
)
print(response.json())

# 3. Indexar documentos
files = {
    'files': open('documento.pdf', 'rb')
}
data = {
    'context': 'meu_contexto',
    'embedding_provider': 'ollama'
}
response = requests.post(
    "http://localhost:8000/api/index",
    files=files,
    data=data
)
print(response.json())

# 4. Fazer pergunta
response = requests.post(
    "http://localhost:8000/api/query",
    json={
        "question": "Qual o horário da piscina?",
        "context": "meu_contexto",
        "llm_provider": "openai",
        "embedding_provider": "ollama",
        "top_k": 8
    }
)
result = response.json()
print("Resposta:", result['answer'])
print("Fontes:", result['sources'])

# 5. Listar contextos
response = requests.get("http://localhost:8000/api/contexts")
print(response.json())
```

### JavaScript (fetch)

```javascript
// 1. Fazer pergunta
async function query(question, context = 'default') {
  const response = await fetch('http://localhost:8000/api/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question: question,
      context: context,
      llm_provider: 'openai',
      embedding_provider: 'ollama',
      top_k: 8,
      return_sources: true
    })
  });

  const data = await response.json();
  console.log('Resposta:', data.answer);
  console.log('Fontes:', data.sources);
  return data;
}

// 2. Upload de arquivo
async function uploadDocument(file, context = 'default') {
  const formData = new FormData();
  formData.append('files', file);
  formData.append('context', context);
  formData.append('embedding_provider', 'ollama');

  const response = await fetch('http://localhost:8000/api/index', {
    method: 'POST',
    body: formData
  });

  return await response.json();
}

// 3. Listar contextos
async function listContexts() {
  const response = await fetch('http://localhost:8000/api/contexts');
  const data = await response.json();
  console.log('Contextos:', data.contexts);
  return data;
}

// Uso
query('Posso ter cachorro?', 'cond_391');
```

### cURL (Command Line)

```bash
# Windows PowerShell

# 1. Health check
curl http://localhost:8000/health

# 2. Criar contexto
curl -X POST http://localhost:8000/api/contexts ^
  -H "Content-Type: application/json" ^
  -d "{\"name\": \"cond_391\"}"

# 3. Upload de arquivo
curl -X POST http://localhost:8000/api/index ^
  -F "files=@regimento.pdf" ^
  -F "context=cond_391" ^
  -F "embedding_provider=ollama"

# 4. Fazer pergunta
curl -X POST http://localhost:8000/api/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"Posso ter cachorro?\", \"context\": \"cond_391\", \"llm_provider\": \"openai\", \"embedding_provider\": \"ollama\", \"top_k\": 8}"

# 5. Listar contextos
curl http://localhost:8000/api/contexts

# 6. Stats de contexto
curl http://localhost:8000/api/contexts/cond_391/stats

# 7. Deletar contexto
curl -X DELETE http://localhost:8000/api/contexts/cond_392
```

### Postman

1. **Importar coleção**: Acesse `http://localhost:8000/docs` e clique em "Download OpenAPI spec"
2. **Criar request**: New Request → POST → `http://localhost:8000/api/query`
3. **Body**: Selecione "raw" → "JSON"
4. **Exemplo**:
   ```json
   {
     "question": "Qual o horário da piscina?",
     "context": "cond_391",
     "llm_provider": "openai"
   }
   ```

---

## Integração Docker

### Dockerfile

O [Dockerfile](Dockerfile) já suporta a API:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copia requirements e instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia código
COPY . .

# Expõe portas
EXPOSE 7860 8000

# Comando padrão (Gradio)
CMD ["python", "app.py"]
```

### docker-compose.yml

Adicione serviço para a API:

```yaml
version: '3.8'

services:
  # Interface Gradio
  rag-gradio:
    build: .
    container_name: rag-gradio
    ports:
      - "7860:7860"
    volumes:
      - ./data/faiss_index:/app/data/faiss_index
      - ./data/documents:/app/data/documents
    env_file:
      - .env
    command: python app.py
    extra_hosts:
      - "host.docker.internal:host-gateway"

  # API REST
  rag-api:
    build: .
    container_name: rag-api
    ports:
      - "8000:8000"
    volumes:
      - ./data/faiss_index:/app/data/faiss_index
      - ./data/documents:/app/data/documents
      - ./data/temp_uploads:/app/data/temp_uploads
    env_file:
      - .env
    command: uvicorn src.api:app --host 0.0.0.0 --port 8000
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

**Executar ambos:**
```bash
docker-compose up -d
```

**Acessar:**
- Gradio: http://localhost:7860
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## Tratamento de Erros

A API usa códigos HTTP padrão e retorna mensagens de erro claras:

### 400 - Bad Request

Requisição inválida (parâmetros faltando ou incorretos)

```json
{
  "detail": "Pergunta vazia."
}
```

### 404 - Not Found

Recurso não encontrado (contexto, índice)

```json
{
  "detail": "Contexto 'cond_999' não encontrado. Use /api/contexts para ver contextos disponíveis."
}
```

### 409 - Conflict

Conflito (contexto já existe)

```json
{
  "detail": "Contexto 'cond_391' já existe."
}
```

### 500 - Internal Server Error

Erro interno do servidor

```json
{
  "detail": "Erro ao processar query: [detalhes do erro]"
}
```

### Tratamento no Cliente

```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/api/query",
        json={"question": "..."}
    )
    response.raise_for_status()  # Levanta exceção se status >= 400
    data = response.json()
    print(data['answer'])

except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        print("Contexto não encontrado")
    elif e.response.status_code == 400:
        print("Requisição inválida:", e.response.json()['detail'])
    else:
        print("Erro:", e)

except requests.exceptions.ConnectionError:
    print("API não está rodando")
```

---

## Segurança

### CORS

Por padrão, CORS está habilitado para **todas as origens**:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Produção: especificar origens
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Para produção**, restrinja origens:

```python
allow_origins=[
    "https://seu-frontend.com",
    "https://app.exemplo.com",
]
```

### API Keys

As API keys (OpenAI, Anthropic) são lidas de variáveis de ambiente:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**Nunca exponha as keys no código ou commits!**

### Autenticação (Futura)

Para adicionar autenticação à API:

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/api/query")
def query(
    req: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Validar token
    if credentials.credentials != "seu-token-secreto":
        raise HTTPException(status_code=401, detail="Não autorizado")

    # Processar query...
```

### Rate Limiting (Futura)

Para limitar requisições:

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/query")
@limiter.limit("10/minute")  # 10 req/min por IP
def query(request: Request, req: QueryRequest):
    # ...
```

---

## Performance

### Otimizações

1. **Caching de embeddings**: Evita reprocessar mesma pergunta
2. **Batch processing**: Upload múltiplos arquivos de uma vez
3. **TOON format**: Economia de 30-60% em tokens
4. **FAISS**: Busca ultra-rápida (<50ms)

### Benchmarks

| Endpoint | Tempo Médio | Notas |
|----------|-------------|-------|
| `/health` | <10ms | Simples check |
| `/api/contexts` | <50ms | Lista contextos |
| `/api/query` | 3-6s | Inclui LLM (2-5s) |
| `/api/index` | ~0.5s/chunk | Depende do tamanho |

### Escalabilidade

Para alta carga:

1. **Múltiplas workers**:
   ```bash
   uvicorn src.api:app --workers 4
   ```

2. **Gunicorn + Uvicorn**:
   ```bash
   gunicorn src.api:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

3. **Load balancer**: NGINX na frente

4. **Cache Redis**: Para queries frequentes

---

## Monitoramento

### Logs

Uvicorn loga automaticamente:

```
INFO:     127.0.0.1:52341 - "POST /api/query HTTP/1.1" 200 OK
INFO:     127.0.0.1:52342 - "GET /api/contexts HTTP/1.1" 200 OK
```

### Healthcheck Avançado

```python
@app.get("/health/detailed")
def detailed_health():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "contexts": len(_get_context_manager().list_contexts()),
        "ollama_connected": check_ollama(),
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "disk_space_gb": get_disk_space(),
    }
```

---

## Próximos Passos

Recursos implementados:
- [x] Histórico de conversas
- [x] Hierarquia legal com busca em cascata
- [x] Perfis de acesso (anônimo, autenticado, admin)
- [x] Motor de verificação de documentos
- [x] Prompt anônimo customizável

Recursos planejados:
- [ ] Autenticação JWT
- [ ] Rate limiting
- [ ] WebSocket para streaming de respostas
- [ ] Cache de queries frequentes (Redis)
- [ ] Métricas (Prometheus)
- [ ] Logs estruturados (JSON)
- [ ] Swagger UI customizado
- [ ] Gerenciamento de conversas (criar, listar, snapshots)

---

## Conclusão

A API REST fornece uma interface completa e profissional para integração do sistema RAG com:

- ✅ Endpoints RESTful bem definidos (15 endpoints)
- ✅ Validação automática de dados
- ✅ Documentação interativa (Swagger)
- ✅ Suporte a múltiplos contextos com hierarquia legal
- ✅ Busca em cascata com score threshold
- ✅ Perfis de acesso e governança
- ✅ Motor de verificação de documentos
- ✅ Histórico de conversa integrado
- ✅ Flexibilidade de provedores (LLM, embeddings)
- ✅ CORS habilitado
- ✅ Tratamento de erros robusto
- ✅ Production-ready

**Comece agora:**
```bash
python -m uvicorn src.api:app --reload
```

Acesse http://localhost:8000/docs e explore!

---

## Referências

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [COMO_FUNCIONA.md](COMO_FUNCIONA.md) - Documentação completa do sistema
- [src/api.py](src/api.py) - Código-fonte da API
