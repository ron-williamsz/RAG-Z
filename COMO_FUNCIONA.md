# Como Funciona o Sistema RAG

## Índice
1. [Visão Geral](#visão-geral)
2. [Arquitetura do Sistema](#arquitetura-do-sistema)
3. [Fluxo de Funcionamento](#fluxo-de-funcionamento)
4. [Tecnologias Utilizadas](#tecnologias-utilizadas)
5. [Componentes Principais](#componentes-principais)
6. [Processo de Indexação](#processo-de-indexação)
7. [Processo de Consulta](#processo-de-consulta)
8. [Sistema de Embeddings](#sistema-de-embeddings)
9. [Armazenamento Vetorial](#armazenamento-vetorial)
10. [Geração de Respostas](#geração-de-respostas)
11. [Múltiplos Contextos](#múltiplos-contextos)

---

## Visão Geral

Este é um **sistema RAG (Retrieval-Augmented Generation)** completo que permite:
- Indexar documentos de diversos formatos (PDF, DOCX, XLSX, TXT, MD)
- Realizar buscas semânticas inteligentes
- Gerar respostas contextualizadas usando LLMs (GPT-4o, Claude)
- Gerenciar múltiplos contextos independentes
- Operar de forma local e privada

### O que é RAG?

RAG combina três capacidades:
1. **Retrieval (Recuperação)**: Busca informações relevantes em documentos
2. **Augmentation (Aumento)**: Enriquece o prompt do LLM com contexto específico
3. **Generation (Geração)**: Produz respostas baseadas no contexto recuperado

---

## Arquitetura do Sistema

```
┌─────────────────┐
│   INTERFACE     │
│    (Gradio)     │ ← Interface web para usuário
└────────┬────────┘
         │
┌────────▼────────────────────────────────────────┐
│              CAMADA DE ORQUESTRAÇÃO             │
│                   (app.py)                      │
│  - Gerenciamento de contextos                  │
│  - Controle de fluxo                            │
│  - Integração de componentes                   │
└────────┬────────────────────────────────────────┘
         │
    ┌────┴────┬──────────┬──────────┬─────────┐
    │         │          │          │         │
┌───▼───┐ ┌──▼──┐  ┌────▼────┐ ┌──▼───┐ ┌───▼────┐
│ DOC   │ │CHUNK│  │EMBEDDINGS│ │VECTOR│ │  RAG   │
│LOADER │ │ER   │  │ MANAGER  │ │STORE │ │ CHAIN  │
└───────┘ └─────┘  └──────────┘ └──────┘ └────────┘
    │         │          │          │         │
    └─────────┴──────────┴──────────┴─────────┘
                      │
              ┌───────▼────────┐
              │  PERSISTÊNCIA  │
              │  (FAISS Index) │
              │  data/faiss_   │
              │    index/      │
              └────────────────┘
```

---

## Fluxo de Funcionamento

### 1. Fase de Indexação

```mermaid
graph LR
    A[Upload Documento] --> B[Document Loader]
    B --> C[Chunker]
    C --> D[Embeddings BGE-M3]
    D --> E[FAISS Index]
    E --> F[Persistência em Disco]
```

**Passo a passo:**

1. **Upload**: Usuário envia documento via interface Gradio
2. **Carregamento**: `DocumentLoader` extrai texto e metadados
3. **Chunking**: `Chunker` divide texto em blocos de 512 caracteres com overlap de 50
4. **Embedding**: Cada chunk é convertido em vetor de 1024 dimensões pelo BGE-M3
5. **Indexação**: Vetores são armazenados no índice FAISS
6. **Persistência**: Índice é salvo em disco em `data/faiss_index/`

### 2. Fase de Consulta

```mermaid
graph LR
    A[Pergunta do Usuário] --> B[Embedding BGE-M3]
    B --> C[Busca FAISS]
    C --> D[Top-K Chunks]
    D --> E[Formatação TOON]
    E --> F[LLM GPT-4o]
    F --> G[Resposta + Fontes]
```

**Passo a passo:**

1. **Pergunta**: Usuário digita pergunta na interface
2. **Embedding**: Pergunta é convertida em vetor de 1024D
3. **Busca Semântica**: FAISS encontra os 8 chunks mais similares
4. **Filtragem**: Apenas chunks com similaridade > 0.7 são mantidos
5. **Formatação**: Contexto é comprimido em formato TOON (economia de 30-60% tokens)
6. **Prompt**: Sistema monta prompt com contexto + pergunta
7. **LLM**: GPT-4o gera resposta completa citando fontes
8. **Retorno**: Resposta + lista de fontes é exibida ao usuário

---

## Tecnologias Utilizadas

### Stack Completo

| Camada | Tecnologia | Versão | Função |
|--------|-----------|--------|--------|
| **Framework RAG** | LangChain | 0.1.0+ | Orquestração de componentes |
| **Vector Database** | FAISS (CPU) | 1.7.4+ | Busca vetorial ultra-rápida |
| **Embeddings** | Ollama BGE-M3 | - | Conversão texto → vetor (local) |
| **LLMs** | OpenAI GPT-4o | - | Geração de respostas |
| | Anthropic Claude Sonnet 4 | - | Alternativa de LLM |
| **Interface** | Gradio | 4.0+ | Interface web interativa |
| **Compressão** | TOON | 0.1.0 | Economia de tokens |
| **Container** | Docker | - | Ambiente isolado |
| **Config** | TOML | - | Configuração centralizada |

### Modelos de IA

**Embeddings:**
- **BGE-M3** (BAAI/bge-m3)
  - 566M parâmetros
  - 1024 dimensões
  - 8192 tokens de contexto
  - Multilíngue (excelente para português)
  - 100% local via Ollama

**LLMs disponíveis:**
- **GPT-4o** (OpenAI) - Principal, mais preciso
- **Claude Sonnet 4** (Anthropic) - Alternativa
- **Llama** (via OpenRouter) - Opção adicional

---

## Componentes Principais

### 1. Document Loader ([document_loader.py](src/document_loader.py))

**Função:** Carregar e extrair texto de diferentes formatos de arquivo

**Formatos suportados:**
- **PDF**: PyPDFLoader com fallback OCR (Tesseract)
- **DOCX/DOC**: Docx2txtLoader
- **XLSX/XLS**: UnstructuredExcelLoader
- **TXT**: TextLoader
- **MD (Markdown)**: TextLoader

**Saída:**
```python
Document(
    page_content="Texto extraído do documento...",
    metadata={
        "source": "caminho/arquivo.pdf",
        "page": 1,
        "file_type": "pdf"
    }
)
```

### 2. Chunker ([chunker.py](src/chunker.py))

**Função:** Dividir documentos em pedaços menores para melhor processamento

**Configurações:**
- `chunk_size`: 512 caracteres (padrão)
- `chunk_overlap`: 50 caracteres (10% para manter contexto)
- `separators`: `["\n\n", "\n", ". ", " ", ""]` (ordem de preferência)

**Por que chunking?**
- LLMs têm limite de contexto
- Chunks menores = busca mais precisa
- Overlap mantém continuidade entre chunks

**Exemplo:**
```
Documento: "Este é um documento muito longo..."
          ↓
Chunk 1: "Este é um documento muito longo sobre regras..."
Chunk 2: "...sobre regras do condomínio. Não é permitido..."
Chunk 3: "...permitido animais de grande porte..."
```

### 3. Embeddings Manager ([embeddings.py](src/embeddings.py))

**Função:** Converter texto em vetores numéricos (embeddings)

**Provedores disponíveis:**
- **Ollama** (padrão): BGE-M3 local, gratuito, privado
- **OpenAI**: text-embedding-3-small (1536D, pago)
- **HuggingFace**: Vários modelos via API

**Como funciona:**
```python
# Texto
"Posso ter cachorro no condomínio?"

# Embedding (simplificado)
[0.25, -0.85, 0.47, ..., 0.11]  # 1024 números
```

**Vantagens do BGE-M3:**
- ✅ Custo: $0 (local)
- ✅ Privacidade: dados não saem do computador
- ✅ Português: excelente suporte multilíngue
- ✅ Performance: ~0.5s por embedding
- ✅ Offline: funciona sem internet

### 4. Vector Store ([vector_store.py](src/vector_store.py))

**Função:** Armazenar e buscar vetores eficientemente usando FAISS

**FAISS (Facebook AI Similarity Search):**
- Biblioteca otimizada para busca de vizinhos mais próximos
- Busca em <50ms mesmo com milhares de vetores
- Índice persistido em disco

**Estrutura de arquivos:**
```
data/faiss_index/{context}/
├── index.faiss    ← Vetores FAISS (~700KB por 1k chunks)
├── index.pkl      ← Metadados Python (~70KB por 1k chunks)
└── metadata.json  ← Informações do contexto
```

**Operações principais:**
- `add_documents()`: Adicionar novos chunks ao índice
- `similarity_search()`: Buscar chunks similares
- `save()`: Salvar índice em disco
- `load()`: Carregar índice do disco

### 5. RAG Chain ([rag_chain.py](src/rag_chain.py))

**Função:** Orquestrar todo o pipeline RAG

**Pipeline completo:**
```python
1. Recebe pergunta do usuário
2. Gera embedding da pergunta
3. Busca chunks similares no FAISS
4. Formata contexto em TOON
5. Monta prompt com contexto + pergunta
6. Envia para LLM (GPT-4o)
7. Retorna resposta + fontes
```

**Template de prompt:**
```
Você é assistente especializado em {system_context}.

DOCUMENTOS DISPONÍVEIS:
{context}  ← Chunks recuperados do FAISS

PERGUNTA: {question}

INSTRUÇÕES:
- Analise com atenção o contexto
- Seja completo e útil
- CITE AS FONTES sempre
- Contextualize quando possível
```

### 6. TOON Formatter ([toon_formatter.py](src/toon_formatter.py))

**Função:** Compactar contexto para economizar tokens (30-60%)

**Exemplo de economia:**

**Sem TOON (JSON padrão - 180 tokens):**
```json
{
  "sources": [
    {
      "id": 1,
      "content": "Não é permitido ter animais de grande porte...",
      "metadata": {
        "file": "regimento_interno.pdf",
        "chunk": 5
      }
    }
  ]
}
```

**Com TOON (70 tokens - 61% economia):**
```
sources:[{id:1,content:"Não é permitido ter animais...",file:"regimento.pdf",chunk:5}]
```

**Benefícios:**
- Reduz custo da API OpenAI
- Permite incluir mais contexto
- Respostas mais rápidas

### 7. Context Manager ([context_manager.py](src/context_manager.py))

**Função:** Gerenciar múltiplos índices independentes

**Casos de uso:**
- Diferentes condomínios
- Diferentes empresas
- Diferentes projetos

**Operações:**
- Criar novo contexto
- Listar contextos disponíveis
- Trocar contexto ativo
- Deletar contexto
- Limpar índice de um contexto

---

## Processo de Indexação

### Fluxo Detalhado

```
┌─────────────────┐
│ 1. UPLOAD       │
│ documento.pdf   │
└────────┬────────┘
         │
┌────────▼────────────────────────┐
│ 2. DOCUMENT LOADER              │
│ PyPDFLoader extrai texto        │
│ ↓                               │
│ "Este é o regimento interno..." │
│ metadata: {source, page, type}  │
└────────┬────────────────────────┘
         │
┌────────▼─────────────────────────┐
│ 3. CHUNKER                       │
│ Divide em chunks de 512 chars   │
│ ↓                                │
│ Chunk 1: "Este é o regimento..." │
│ Chunk 2: "...interno. Artigo..." │
│ Chunk 3: "...Artigo 1: Não é..." │
│ Total: 15 chunks                 │
└────────┬─────────────────────────┘
         │
┌────────▼──────────────────────────┐
│ 4. EMBEDDINGS (BGE-M3)            │
│ Converte cada chunk em vetor     │
│ ↓                                 │
│ Chunk 1 → [0.25, -0.85, ..., 0.11]│
│ Chunk 2 → [0.13, 0.67, ..., -0.44]│
│ Chunk 3 → [-0.89, 0.22, ..., 0.77]│
│ (Vetores de 1024 dimensões)      │
└────────┬──────────────────────────┘
         │
┌────────▼───────────────────────────┐
│ 5. FAISS INDEXAÇÃO                 │
│ Adiciona vetores ao índice        │
│ Cria estrutura para busca rápida  │
└────────┬───────────────────────────┘
         │
┌────────▼──────────────────────────┐
│ 6. PERSISTÊNCIA                   │
│ Salva em disco:                   │
│ - data/faiss_index/default/       │
│   ├── index.faiss (vetores)       │
│   ├── index.pkl (metadados)       │
│   └── metadata.json (info)        │
└───────────────────────────────────┘
```

### Código Simplificado

```python
# 1. Carregar documento
documents = DocumentLoader.load("documento.pdf")

# 2. Dividir em chunks
chunks = Chunker.split(documents,
                       chunk_size=512,
                       overlap=50)

# 3. Gerar embeddings
embeddings = EmbeddingsManager.embed_documents(chunks)

# 4. Adicionar ao FAISS
vector_store = FAISSVectorStore()
vector_store.add_documents(chunks, embeddings)

# 5. Salvar
vector_store.save("data/faiss_index/default")
```

### Configurações de Indexação

Em [config.toml](config.toml):

```toml
[chunking]
chunk_size = 512        # Tamanho ideal para BGE-M3
chunk_overlap = 50      # Mantém contexto entre chunks
separators = ["\n\n", "\n", ". ", " ", ""]

[embeddings]
provider = "ollama"     # Local, gratuito
model = "bge-m3"        # 1024D, multilíngue
base_url = "http://host.docker.internal:11434"
```

---

## Processo de Consulta

### Fluxo Detalhado

```
┌──────────────────────────────┐
│ 1. PERGUNTA DO USUÁRIO       │
│ "Posso ter cachorro?"        │
└────────┬─────────────────────┘
         │
┌────────▼─────────────────────────┐
│ 2. EMBEDDING DA PERGUNTA         │
│ BGE-M3 converte em vetor        │
│ ↓                                │
│ [0.67, -0.23, 0.88, ..., -0.15] │
│ (vetor de 1024 dimensões)       │
└────────┬─────────────────────────┘
         │
┌────────▼────────────────────────────┐
│ 3. BUSCA SEMÂNTICA (FAISS)          │
│ Compara com todos os chunks        │
│ Retorna top_k=8 mais similares     │
│ ↓                                   │
│ Chunk #42: score 0.92 ✓            │
│ Chunk #15: score 0.87 ✓            │
│ Chunk #23: score 0.83 ✓            │
│ Chunk #8:  score 0.78 ✓            │
│ Chunk #31: score 0.74 ✓            │
│ Chunk #19: score 0.71 ✓            │
│ Chunk #5:  score 0.68 ✗ (< 0.7)   │
│ Chunk #12: score 0.65 ✗ (< 0.7)   │
│ → 6 chunks selecionados            │
└────────┬────────────────────────────┘
         │
┌────────▼───────────────────────────┐
│ 4. FORMATAÇÃO CONTEXTO (TOON)     │
│ Compacta chunks para economizar   │
│ ↓                                  │
│ sources:[                          │
│   {id:1, content:"Não é           │
│    permitido animais...",          │
│    file:"regimento.pdf", chunk:42},│
│   {id:2, content:"Artigo 15...",  │
│    file:"regimento.pdf", chunk:15},│
│   ...                              │
│ ]                                  │
│ Economia: 45% tokens              │
└────────┬───────────────────────────┘
         │
┌────────▼────────────────────────────┐
│ 5. MONTAGEM DO PROMPT               │
│ System + Contexto + Pergunta       │
│ ↓                                   │
│ "Você é especialista em documentos │
│  de condomínio.                    │
│                                     │
│  DOCUMENTOS:                        │
│  [contexto formatado em TOON]      │
│                                     │
│  PERGUNTA:                          │
│  Posso ter cachorro?"              │
└────────┬────────────────────────────┘
         │
┌────────▼─────────────────────────────┐
│ 6. GERAÇÃO LLM (GPT-4o)              │
│ OpenAI API processa prompt          │
│ ↓                                    │
│ "De acordo com o Regimento Interno  │
│  (regimento.pdf), não é permitido   │
│  ter animais de grande porte. Para  │
│  animais de pequeno porte como      │
│  cachorros pequenos, é necessário   │
│  aprovação prévia da administração. │
│                                      │
│  Fontes:                             │
│  - regimento.pdf (Artigo 15)        │
│  - regimento.pdf (Artigo 22)"       │
└────────┬─────────────────────────────┘
         │
┌────────▼───────────────────────┐
│ 7. RETORNO AO USUÁRIO          │
│ Resposta + Fontes + Scores     │
└────────────────────────────────┘
```

### Código Simplificado

```python
# 1. Embedding da pergunta
query = "Posso ter cachorro?"
query_vector = embeddings.embed_query(query)

# 2. Busca no FAISS
results = vector_store.similarity_search(
    query_vector,
    top_k=8,
    score_threshold=0.7
)
# → [(chunk, score), (chunk, score), ...]

# 3. Formata contexto
context = toon_formatter.format(results)

# 4. Monta prompt
prompt = f"""
Você é especialista em documentos de condomínio.

DOCUMENTOS:
{context}

PERGUNTA: {query}
"""

# 5. Gera resposta
response = llm.invoke(prompt)

# 6. Retorna
return {
    "answer": response,
    "sources": [r[0].metadata for r in results]
}
```

### Configurações de Consulta

Em [config.toml](config.toml):

```toml
[retrieval]
top_k = 8               # Número de chunks a recuperar
score_threshold = 0.7   # Mínimo de similaridade (0-1)

[llm.openai]
model = "gpt-4o"        # Modelo mais avançado
temperature = 0.3       # Mais conservador/factual
max_tokens = 4096       # Respostas completas
```

---

## Sistema de Embeddings

### O que são Embeddings?

Embeddings são representações numéricas de texto que capturam significado semântico.

**Exemplo:**
```
Texto 1: "cachorro"     → [0.8, 0.2, 0.6, ...]
Texto 2: "cão"          → [0.79, 0.21, 0.59, ...] ← Muito similar!
Texto 3: "computador"   → [-0.3, 0.9, -0.1, ...] ← Diferente
```

Textos com significados similares têm vetores próximos.

### BGE-M3: Nosso Modelo de Embeddings

**Especificações:**
- **Nome completo**: BAAI/bge-m3
- **Parâmetros**: 566 milhões
- **Dimensões**: 1024
- **Contexto máximo**: 8192 tokens (~6000 palavras)
- **Arquitetura**: BERT-based
- **Treinamento**: Multilíngue (103 idiomas)
- **Tamanho**: 1.2 GB em disco

**Por que BGE-M3?**

| Aspecto | BGE-M3 | OpenAI text-embedding-3-small |
|---------|--------|-------------------------------|
| **Custo** | $0 (local) | $0.02 por 1M tokens |
| **Privacidade** | 100% local | Dados enviados à OpenAI |
| **Português** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Velocidade** | ~0.5s | ~0.2s (+ latência rede) |
| **Dimensões** | 1024 | 1536 |
| **Offline** | ✅ Sim | ❌ Não |

**Performance em Português:**
- Top 3 mundial em benchmarks multilíngues
- Treinado específicamente para retrieval
- Excelente com documentos técnicos/legais

### Como Funciona (Técnico)

```python
# 1. Tokenização
texto = "Não é permitido animais"
tokens = ["Não", "é", "permitido", "animais"]

# 2. Conversão para IDs
token_ids = [1234, 5678, 9012, 3456]

# 3. Passagem pela rede neural
# (566M parâmetros processam os tokens)

# 4. Embedding final (1024 números)
embedding = [
    0.234, -0.567, 0.891, ..., -0.123
]  # 1024 dimensões

# 5. Normalização
# Vetor é normalizado para ter magnitude 1
# Facilita cálculo de similaridade
```

### Cálculo de Similaridade

Usamos **similaridade de cosseno**:

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """
    Retorna valor entre -1 e 1
    1 = idênticos
    0 = não relacionados
    -1 = opostos
    """
    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )

# Exemplo
pergunta = embed("Posso ter cachorro?")
chunk1 = embed("Não é permitido animais de grande porte")
chunk2 = embed("O horário da piscina é das 8h às 20h")

sim1 = cosine_similarity(pergunta, chunk1)  # 0.92 ← Alta!
sim2 = cosine_similarity(pergunta, chunk2)  # 0.23 ← Baixa
```

### Integração com Ollama

```yaml
# docker-compose.yml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

Permite container Docker acessar Ollama rodando no host:
- URL: `http://host.docker.internal:11434`
- Endpoint: `/api/embeddings`
- Modelo: `bge-m3`

**Requisição exemplo:**
```bash
curl http://localhost:11434/api/embeddings -d '{
  "model": "bge-m3",
  "prompt": "Posso ter cachorro?"
}'
```

**Resposta:**
```json
{
  "embedding": [0.234, -0.567, 0.891, ..., -0.123]
}
```

---

## Armazenamento Vetorial

### FAISS: Facebook AI Similarity Search

**O que é?**
- Biblioteca C++ otimizada para busca de vizinhos mais próximos
- Desenvolvida pelo Facebook AI Research
- Extremamente rápida mesmo com milhões de vetores

**Por que FAISS?**
- ✅ Busca em <50ms mesmo com 100k vetores
- ✅ Funciona em CPU (não precisa GPU)
- ✅ Índice persistido em disco
- ✅ Baixo uso de memória
- ✅ Open source e amplamente usado

### Estrutura do Índice

```
data/faiss_index/{context}/
│
├── index.faiss          ← Índice binário FAISS
│   │                      Contém: vetores + estrutura de busca
│   │                      Tamanho: ~700KB por 1000 chunks
│   └── Formato: binário otimizado
│
├── index.pkl            ← Metadados Python (pickle)
│   │                      Contém: textos originais, metadados
│   │                      Tamanho: ~70KB por 1000 chunks
│   └── Formato: Python pickle
│
└── metadata.json        ← Informações do contexto
    │                      Contém: lista de arquivos, timestamps
    │                      Tamanho: ~1-5KB
    └── Formato: JSON legível
```

### Como Funciona a Busca FAISS

```python
# 1. Indexação (uma vez)
index = faiss.IndexFlatL2(1024)  # 1024 dimensões
index.add(embeddings)             # Adiciona vetores
faiss.write_index(index, "index.faiss")

# 2. Busca (a cada consulta)
query_vector = embed("Posso ter cachorro?")
distances, indices = index.search(
    query_vector,
    k=8  # top 8 resultados
)

# Resultado:
# distances: [0.08, 0.13, 0.17, 0.22, 0.26, 0.29, 0.32, 0.38]
# indices:   [42,   15,   23,   8,    31,   19,   5,    12]
#             ↑ Chunk #42 é o mais similar (distância 0.08)
```

**Conversão distância → similaridade:**
```python
# FAISS retorna distância L2 (menor = mais similar)
# Convertemos para score de similaridade (maior = mais similar)

similarity = 1 / (1 + distance)

# Exemplo:
distance = 0.08  → similarity = 0.926 (92.6%)
distance = 0.38  → similarity = 0.725 (72.5%)
```

### Tipos de Índice FAISS

Atualmente usamos **IndexFlatL2** (busca exata):

| Tipo | Precisão | Velocidade | Memória | Uso |
|------|----------|-----------|---------|-----|
| **IndexFlatL2** | 100% | Média | Alta | Até ~100k vetores |
| IndexIVFFlat | 95-99% | Rápida | Média | 100k-1M vetores |
| IndexHNSW | 95-99% | Muito rápida | Alta | 1M+ vetores |

Para projetos maiores, pode-se migrar para IndexIVFFlat ou IndexHNSW.

### Persistência

**Dados são salvos localmente:**
```
./data/faiss_index/  ← Volume Docker montado
```

**Dados permanecem quando:**
- ✅ Container reinicia
- ✅ Docker Compose down/up (sem -v)
- ✅ Computador reinicia
- ✅ Código é atualizado

**Dados perdidos se:**
- ❌ Deletar pasta `data/faiss_index/`
- ❌ `docker-compose down -v` (flag -v remove volumes)
- ❌ Deletar arquivos `index.faiss` ou `index.pkl`

**Backup recomendado:**
```powershell
# Windows PowerShell
Copy-Item -Recurse ./data/faiss_index ./backup_$(Get-Date -Format 'yyyyMMdd')

# Restaurar
Copy-Item -Recurse ./backup_20251215/* ./data/faiss_index/
```

---

## Geração de Respostas

### LLMs Disponíveis

#### 1. GPT-4o (OpenAI) - Padrão

**Especificações:**
- **Modelo**: gpt-4o
- **Contexto**: 128k tokens
- **Custo**: ~$0.005/1k input, ~$0.015/1k output
- **Velocidade**: 2-5s por resposta

**Configuração:**
```toml
[llm.openai]
model = "gpt-4o"
api_key = "${OPENAI_API_KEY}"  # Variável de ambiente
temperature = 0.3               # Mais conservador
max_tokens = 4096               # Respostas completas
```

**Vantagens:**
- ✅ Muito preciso e confiável
- ✅ Excelente em português
- ✅ Segue instruções rigorosamente
- ✅ Boa citação de fontes

#### 2. Claude Sonnet 4 (Anthropic) - Alternativa

**Especificações:**
- **Modelo**: claude-sonnet-4-20250514
- **Contexto**: 200k tokens
- **Custo**: Similar ao GPT-4o
- **Velocidade**: 2-5s por resposta

**Configuração:**
```toml
[llm.anthropic]
model = "claude-sonnet-4-20250514"
api_key = "${ANTHROPIC_API_KEY}"
temperature = 0.3
max_tokens = 4096
```

**Vantagens:**
- ✅ Respostas mais naturais
- ✅ Maior janela de contexto
- ✅ Excelente raciocínio

#### 3. Llama (OpenRouter) - Econômico

**Configuração:**
```toml
[llm.openrouter]
model = "meta-llama/llama-3.2-90b-vision-instruct"
api_key = "${OPENROUTER_API_KEY}"
```

### Pipeline de Geração

```python
# 1. Template de prompt
PROMPT_TEMPLATE = """
Você é assistente especializado em {system_context}.

DOCUMENTOS DISPONÍVEIS:
{context}

PERGUNTA: {question}

INSTRUÇÕES:
- Analise com atenção o contexto fornecido
- Seja completo e útil na resposta
- CITE AS FONTES sempre que usar informação dos documentos
- Contextualize a resposta quando apropriado
- Seja proativo com informações relacionadas
- Organize bem a resposta
"""

# 2. Montagem do prompt
prompt = PROMPT_TEMPLATE.format(
    system_context="documentos de condomínio",
    context=toon_formatted_context,
    question=user_question
)

# 3. Chamada ao LLM
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Você é um assistente..."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.3,
    max_tokens=4096
)

# 4. Extração da resposta
answer = response.choices[0].message.content
```

### Customização do System Context

Pode-se personalizar o comportamento por contexto:

```python
# Para condomínio
system_context = """
documentos de condomínio. Você deve:
- Sempre citar artigos e números quando disponíveis
- Explicar procedimentos administrativos claramente
- Alertar sobre prazos e penalidades quando relevante
"""

# Para empresa
system_context = """
documentação interna da empresa. Você deve:
- Ser direto e objetivo
- Focar em procedimentos práticos
- Incluir contatos relevantes quando apropriado
"""
```

### Controle de Temperature

**Temperature**: Controla aleatoriedade/criatividade

```python
# Temperature = 0.0 (Determinístico)
# - Sempre a mesma resposta
# - Muito conservador
# - Ideal para: documentos legais

# Temperature = 0.3 (Padrão do sistema)
# - Consistente mas com variação mínima
# - Bom balanço precisão/naturalidade
# - Ideal para: RAG geral

# Temperature = 0.7 (Criativo)
# - Mais variação nas respostas
# - Mais natural mas menos preciso
# - Ideal para: chatbots gerais

# Temperature = 1.0 (Muito criativo)
# - Alta variação
# - Pode "alucinar" informações
# - Evitar em RAG
```

### Exemplo de Resposta Completa

**Pergunta:**
```
"Posso ter cachorro no condomínio?"
```

**Contexto recuperado (TOON):**
```
sources:[
  {id:1, content:"Artigo 15 - Não é permitido ter animais de grande porte nas unidades...", file:"regimento.pdf", chunk:42, score:0.92},
  {id:2, content:"Artigo 22 - Animais de pequeno porte necessitam aprovação prévia...", file:"regimento.pdf", chunk:15, score:0.87}
]
```

**Resposta do GPT-4o:**
```
De acordo com o Regimento Interno do condomínio:

**Animais de grande porte**: Não são permitidos (Artigo 15, regimento.pdf).

**Animais de pequeno porte** (como cachorros pequenos): São permitidos
mediante aprovação prévia da administração (Artigo 22, regimento.pdf).

**Próximos passos:**
1. Verifique o porte do seu cachorro (consulte a administração sobre
   os critérios)
2. Se for pequeno porte, solicite autorização prévia à administração
3. Aguarde aprovação antes de trazer o animal

**Fontes:**
- regimento.pdf - Artigo 15 (Relevância: 92%)
- regimento.pdf - Artigo 22 (Relevância: 87%)
```

**Características da resposta:**
- ✅ Cita fontes específicas
- ✅ Organizada e estruturada
- ✅ Proativa (próximos passos)
- ✅ Contextualizada
- ✅ Baseada nos documentos

---

## Múltiplos Contextos

### Conceito

O sistema permite criar **contextos independentes**, cada um com seu próprio índice FAISS e documentos.

**Casos de uso:**
- Diferentes condomínios (cada um com seus documentos)
- Diferentes empresas
- Diferentes projetos
- Separação por tipo de documento (contratos, normas, etc.)

### Estrutura de Diretórios

```
data/faiss_index/
│
├── default/                  ← Contexto padrão
│   ├── index.faiss
│   ├── index.pkl
│   └── metadata.json
│
├── condominio_169/           ← Contexto 1
│   ├── index.faiss
│   ├── index.pkl
│   └── metadata.json
│
├── condominio_170/           ← Contexto 2
│   ├── index.faiss
│   ├── index.pkl
│   └── metadata.json
│
└── empresa_xyz/              ← Contexto 3
    ├── index.faiss
    ├── index.pkl
    └── metadata.json
```

### Operações de Contexto

#### 1. Criar Novo Contexto

```python
# Via ContextManager
context_manager.create_context("condominio_169")

# Cria:
# - data/faiss_index/condominio_169/
# - Índice FAISS vazio
# - metadata.json inicial
```

**Na interface Gradio:**
1. Aba "Gerenciamento de Contextos"
2. Digite nome do contexto
3. Clique "Criar Novo Contexto"

#### 2. Listar Contextos

```python
contexts = context_manager.list_contexts()
# ['default', 'condominio_169', 'condominio_170', 'empresa_xyz']
```

#### 3. Carregar/Trocar Contexto

```python
# Carrega contexto diferente
context_manager.load_context("condominio_169")

# Agora:
# - Buscas usam índice de condominio_169
# - Indexações vão para condominio_169
# - Isolamento completo
```

**Na interface Gradio:**
1. Dropdown "Selecionar Contexto"
2. Escolhe contexto
3. Sistema automaticamente carrega

#### 4. Deletar Contexto

```python
# Remove contexto e todos os dados
context_manager.delete_context("condominio_169")

# Deleta:
# - data/faiss_index/condominio_169/ (pasta inteira)
# - Todos os índices
# - Todos os metadados
```

⚠️ **ATENÇÃO**: Ação irreversível! Faça backup antes.

#### 5. Limpar Índice

```python
# Mantém contexto mas remove todos os documentos
context_manager.clear_context("condominio_169")

# Resultado:
# - Índice vazio
# - metadata.json resetado
# - Contexto ainda existe
```

Útil para reindexar do zero.

### Metadata.json

Cada contexto tem um arquivo `metadata.json` com informações:

```json
{
  "context_name": "condominio_169",
  "created_at": "2024-01-15T10:30:00",
  "last_updated": "2024-01-20T15:45:00",
  "files": [
    {
      "filename": "regimento_interno.pdf",
      "indexed_at": "2024-01-15T10:35:00",
      "chunks": 45,
      "size_bytes": 524288
    },
    {
      "filename": "ata_assembleia_2024.pdf",
      "indexed_at": "2024-01-20T15:45:00",
      "chunks": 23,
      "size_bytes": 312576
    }
  ],
  "total_chunks": 68,
  "total_files": 2,
  "embeddings_model": "bge-m3",
  "embeddings_provider": "ollama"
}
```

**Informações rastreadas:**
- Nome do contexto
- Timestamps de criação e última atualização
- Lista completa de arquivos indexados
- Número de chunks por arquivo
- Totais agregados
- Modelo de embeddings usado

### Isolamento entre Contextos

```python
# Contexto 1: condominio_169
context_manager.load_context("condominio_169")
response1 = rag_chain.query("Qual o horário da piscina?")
# → Busca APENAS em documentos de condominio_169

# Contexto 2: condominio_170
context_manager.load_context("condominio_170")
response2 = rag_chain.query("Qual o horário da piscina?")
# → Busca APENAS em documentos de condominio_170

# Respostas podem ser diferentes!
```

**Vantagens:**
- ✅ Zero contaminação entre contextos
- ✅ Privacidade de dados
- ✅ Performance (busca em menos documentos)
- ✅ Organização clara

### Fluxo de Trabalho com Múltiplos Contextos

```
1. SETUP INICIAL
   ├─ Criar contexto "condominio_169"
   ├─ Criar contexto "condominio_170"
   └─ Criar contexto "empresa_xyz"

2. INDEXAÇÃO
   ├─ Carregar "condominio_169"
   │  ├─ Upload regimento_169.pdf
   │  ├─ Upload ata_2024_169.pdf
   │  └─ Indexar
   │
   ├─ Carregar "condominio_170"
   │  ├─ Upload regimento_170.pdf
   │  ├─ Upload ata_2024_170.pdf
   │  └─ Indexar
   │
   └─ Carregar "empresa_xyz"
      ├─ Upload manual_funcionarios.pdf
      ├─ Upload politicas.pdf
      └─ Indexar

3. USO DIÁRIO
   ├─ Atender condomínio 169
   │  ├─ Selecionar contexto "condominio_169"
   │  └─ Fazer perguntas
   │
   ├─ Atender condomínio 170
   │  ├─ Selecionar contexto "condominio_170"
   │  └─ Fazer perguntas
   │
   └─ Consultar empresa
      ├─ Selecionar contexto "empresa_xyz"
      └─ Fazer perguntas
```

### Estatísticas por Contexto

```python
stats = context_manager.get_context_stats("condominio_169")

# Retorna:
{
    "name": "condominio_169",
    "total_files": 12,
    "total_chunks": 456,
    "total_size_mb": 8.7,
    "oldest_file": "2024-01-15T10:35:00",
    "newest_file": "2024-12-10T09:20:00",
    "embeddings_model": "bge-m3",
    "avg_chunks_per_file": 38
}
```

Útil para monitoramento e analytics.

---

## Performance e Escalabilidade

### Benchmarks de Velocidade

| Operação | Tempo | Notas |
|----------|-------|-------|
| **Embedding (1 chunk)** | ~0.5s | BGE-M3 via Ollama |
| **Embedding (batch 10)** | ~2s | Paralelização interna |
| **Busca FAISS (1k docs)** | <10ms | Ultra-rápido |
| **Busca FAISS (10k docs)** | <50ms | Escala bem |
| **Busca FAISS (100k docs)** | ~200ms | Ainda rápido |
| **LLM GPT-4o** | 2-5s | Depende tamanho resposta |
| **Query completo (fim a fim)** | 3-6s | Indexado + resposta |

### Limites Práticos

| Recurso | Limite Recomendado | Limite Técnico | Observação |
|---------|-------------------|---------------|------------|
| **Documentos por contexto** | 1.000 | 10.000+ | Performance degrada gradualmente |
| **Chunks totais** | 10.000 | 100.000+ | FAISS suporta milhões |
| **Tamanho de arquivo** | 50 MB | 500 MB | PDF muito grandes travam OCR |
| **Contextos simultâneos** | 10 | 100+ | Limitado por disco |
| **Queries simultâneas** | 5 | 20+ | Limitado por Ollama/OpenAI |

### Otimizações Implementadas

#### 1. TOON Format (30-60% economia)
```python
# Sem TOON: 1000 tokens
# Com TOON: 400-700 tokens
# Economia: $0.015 → $0.006 por query (GPT-4o)
```

#### 2. Batch Embeddings
```python
# Sem batch: 10 chunks × 0.5s = 5s
# Com batch: 10 chunks = 2s (2.5x mais rápido)
embeddings.embed_documents([chunk1, chunk2, ...])
```

#### 3. FAISS CPU Otimizado
```python
# IndexFlatL2: busca exata, rápida até 100k docs
# Pode migrar para IndexIVFFlat se > 100k
```

#### 4. Caching de Contexto
```python
# RAGChain mantém contexto em memória
# Evita recarregar FAISS a cada query
```

### Uso de Recursos

**Memória (Docker container):**
```
Ollama + BGE-M3:   ~1.2 GB
App Python:        ~200 MB
FAISS index (10k): ~100 MB (em memória)
-----------------------------------
Total:             ~1.5 GB
```

**Disco:**
```
Por 1.000 chunks:
- index.faiss: ~700 KB
- index.pkl:   ~70 KB
-----------------------------------
Total:         ~770 KB

Por 10.000 chunks:
- index.faiss: ~7 MB
- index.pkl:   ~700 KB
-----------------------------------
Total:         ~7.7 MB
```

**CPU:**
```
Em repouso:    ~5% (1 core)
Durante embed: ~80-100% (1 core, ~0.5s)
Durante busca: ~20% (1 core, <50ms)
Durante LLM:   ~5% (processamento remoto)
```

### Custos Estimados (OpenAI)

**Por query (média):**
```
Input:  1.500 tokens (prompt + contexto TOON) × $0.005/1k = $0.0075
Output:   500 tokens (resposta)             × $0.015/1k = $0.0075
-------------------------------------------------------------------
Total por query: ~$0.015 (1.5 centavos)

1000 queries/mês: ~$15
10.000 queries/mês: ~$150
```

**Economia com TOON:**
```
Sem TOON: 2.500 tokens input → $0.0125
Com TOON: 1.500 tokens input → $0.0075
-------------------------------------------
Economia: 40% no custo de input
```

### Escalabilidade Futura

**Para crescer além dos limites atuais:**

1. **Embeddings:**
   - Manter BGE-M3 local (escala bem)
   - Ou migrar para GPU (10x mais rápido)

2. **FAISS:**
   - Migrar de IndexFlatL2 para IndexIVFFlat (100k-1M docs)
   - Ou IndexHNSW (1M+ docs, mais memória)

3. **LLM:**
   - Implementar caching de respostas
   - Rate limiting para controlar custos
   - Llama local para reduzir custos

4. **Infraestrutura:**
   - Redis para cache distribuído
   - Load balancer para múltiplas instâncias
   - PostgreSQL para metadados (vs JSON)

---

## Conclusão

Este sistema RAG é uma solução completa e profissional que combina:

- **Embeddings locais** (BGE-M3) para privacidade e custo zero
- **FAISS** para busca vetorial ultra-rápida
- **LLMs de ponta** (GPT-4o, Claude) para respostas precisas
- **Múltiplos contextos** para organização e isolamento
- **TOON format** para economia de tokens (30-60%)
- **Docker** para deploy simples e consistente
- **Interface Gradio** intuitiva e responsiva

O sistema está **production-ready** e pode escalar para milhares de documentos e centenas de queries diárias.

---

## Referências

### Documentação Interna
- [README.md](README.md) - Guia de início rápido
- [ARQUITETURA_RAG.md](ARQUITETURA_RAG.md) - Detalhes sobre BGE-M3
- [ANALISE_BGE_M3.md](ANALISE_BGE_M3.md) - Comparação Ollama vs HuggingFace
- [PERSISTENCIA_DADOS.md](PERSISTENCIA_DADOS.md) - Sistema de backup

### Código-Fonte Principal
- [app.py](app.py) - Interface Gradio e orquestração
- [src/document_loader.py](src/document_loader.py) - Carregamento de documentos
- [src/chunker.py](src/chunker.py) - Divisão em chunks
- [src/embeddings.py](src/embeddings.py) - Geração de embeddings
- [src/vector_store.py](src/vector_store.py) - Armazenamento FAISS
- [src/rag_chain.py](src/rag_chain.py) - Pipeline RAG
- [src/context_manager.py](src/context_manager.py) - Múltiplos contextos
- [src/toon_formatter.py](src/toon_formatter.py) - Compressão TOON

### Configuração
- [config.toml](config.toml) - Todas as configurações do sistema
- [docker-compose.yml](docker-compose.yml) - Orquestração Docker
- [requirements.txt](requirements.txt) - Dependências Python

### Links Externos
- [LangChain](https://python.langchain.com/) - Framework RAG
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - Modelo de embeddings
- [Ollama](https://ollama.ai/) - LLMs locais
- [Gradio](https://gradio.app/) - Interface web
