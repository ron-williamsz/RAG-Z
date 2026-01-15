# Análise do Sistema de Verificação de Documentos

## Visão Geral

Este documento apresenta uma análise detalhada do pipeline de verificação de documentos, identificando os fatores de sucesso da **Etapa 1 (Extração)** e propondo melhorias para a **Etapa 2 (Comparação de Alvos)**.

---

## 1. ANÁLISE DA ETAPA 1 - EXTRAÇÃO DE ENTIDADES

### 1.1 Fluxo Completo do Documento

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ETAPA 1: FLUXO DE EXTRAÇÃO                               │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐
    │   DOCUMENTO     │ (PDF, DOCX, XLSX, TXT, MD)
    │   DE ENTRADA    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ DocumentLoader  │ → Detecção automática de formato
    │                 │ → OCR fallback para PDFs digitalizados
    │                 │ → Enriquecimento de metadados
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │    Chunker      │ → RecursiveCharacterTextSplitter (512 chars)
    │                 │ → Overlap de 50 chars (10%)
    │                 │ → MarkdownHeaderTextSplitter para .md
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  VectorStore    │ → FAISS indexação temporária
    │  (FAISS +       │ → BGE-M3 embeddings (1024 dims, multilingual)
    │   BGE-M3)       │ → Limpeza automática após extração
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ EntityExtractor │ → Cobertura completa (<50 chunks: ALL)
    │    (LLM)        │ → Top-K para documentos grandes
    │                 │ → Temperature = 0.0 (determinístico)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Validação &   │ → Deduplicação case-insensitive
    │   Refinamento   │ → Normalização de capitalização
    │                 │ → Remoção de ruído (códigos, prefixos)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Persistência   │ → SessionManager com TTL
    │  de Sessão      │ → JSON para recuperação
    │                 │ → Source chunks para proveniência
    └─────────────────┘
```

### 1.2 Técnicas de Processamento Utilizadas

| Componente | Técnica | Detalhes |
|------------|---------|----------|
| **OCR** | Tesseract + pdf2image | Fallback automático quando PDF tem <50 chars de texto |
| **Parsing** | LangChain Loaders | PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, TextLoader |
| **Chunking** | RecursiveCharacterTextSplitter | 512 chars, 50 overlap, separadores hierárquicos |
| **Embeddings** | BGE-M3 (Ollama) | 1024 dimensões, multilingual, gratuito |
| **Indexação** | FAISS | Índice temporário por extração |
| **LLM** | GPT-4o / Claude Sonnet | Temperature=0.0 para reprodutibilidade |

### 1.3 Métodos de Validação e Refinamento

#### 1.3.1 Prompt de Extração (Altamente Eficaz)
```python
# Pontos-chave do prompt:
1. Instruções explícitas para ignorar formatação
2. Exemplos concretos de entrada/saída
3. Normalização de capitalização especificada
4. Remoção de duplicatas exigida
5. Output estruturado (uma entidade por linha)
```

#### 1.3.2 Pós-Processamento
```python
# Deduplicação case-insensitive preservando ordem
seen = set()
unique_entities = []
for entity in entities:
    entity_lower = entity.lower()
    if entity_lower not in seen:
        seen.add(entity_lower)
        unique_entities.append(entity)
```

### 1.4 Fatores de Sucesso da Etapa 1

| Fator | Impacto | Descrição |
|-------|---------|-----------|
| **Cobertura Completa** | ⭐⭐⭐⭐⭐ | Usa TODOS os chunks se documento <50 chunks |
| **Temperature 0.0** | ⭐⭐⭐⭐⭐ | Extração determinística e reproduzível |
| **Prompt Estruturado** | ⭐⭐⭐⭐⭐ | Exemplos concretos + instruções claras |
| **BGE-M3 Multilingual** | ⭐⭐⭐⭐ | Suporte nativo a português + acentos |
| **OCR Fallback** | ⭐⭐⭐⭐ | Documentos digitalizados processados |
| **Deduplicação Inteligente** | ⭐⭐⭐⭐ | Case-insensitive com preservação de ordem |
| **Metadados Ricos** | ⭐⭐⭐ | Source chunks para auditoria |

---

## 2. ANÁLISE DA ETAPA 2 - COMPARAÇÃO DE ALVOS

### 2.1 Fluxo Atual

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ETAPA 2: FLUXO DE COMPARAÇÃO (ATUAL)                     │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐
    │  DOCUMENTO(S)   │
    │    ALVO(S)      │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ EntityExtractor │ → Mesmo processo da Etapa 1
    │   (repetido)    │ → Extrai entidades do alvo
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────────────────────────────────────┐
    │                 SemanticMatcher                      │
    │                                                      │
    │   ┌───────────────┐     ┌───────────────┐           │
    │   │ Exact Match   │ ──► │ Vector Pre-   │           │
    │   │ (case-insens.)│     │ filtering     │           │
    │   └───────────────┘     │ (BGE-M3)      │           │
    │                         └───────┬───────┘           │
    │                                 │                   │
    │                                 ▼                   │
    │                         ┌───────────────┐           │
    │                         │ LLM Semantic  │           │
    │                         │ Validation    │           │
    │                         └───────┬───────┘           │
    │                                 │                   │
    │                                 ▼                   │
    │                         ┌───────────────┐           │
    │                         │ Parse JSON    │           │
    │                         │ Response      │           │
    │                         └───────────────┘           │
    └─────────────────────────────────────────────────────┘
             │
             ▼
    ┌─────────────────┐
    │ Agregação de    │ → Status: verified/partial_match/mismatch
    │ Resultados      │ → Overall confidence
    └─────────────────┘
```

### 2.2 Pontos Fracos Identificados na Etapa 2

| Problema | Severidade | Impacto |
|----------|------------|---------|
| **1. Embedding por entidade** | 🔴 Alta | Recalcula embeddings para CADA referência vs TODOS os alvos (O(n×m)) |
| **2. Sem normalização prévia** | 🔴 Alta | Variações de acentos/case comparadas diretamente |
| **3. Threshold fixo** | 🟡 Média | Strictness não considera tipos de entidade |
| **4. Sem cache de embeddings** | 🟡 Média | Recalcula mesmo embeddings repetidos |
| **5. LLM para cada par** | 🔴 Alta | Uma chamada LLM por entidade de referência |
| **6. Sem validação cruzada** | 🟡 Média | Entidade A→B não verifica B→A |
| **7. Regex JSON frágil** | 🟡 Média | `r"\{[^}]+\}"` falha com JSON aninhado |

### 2.3 Análise de Performance

```
CENÁRIO: 100 entidades de referência vs documento alvo com 100 entidades

ATUAL:
- Embeddings gerados: 100 × 100 = 10.000 operações de embedding
- Chamadas LLM: 100 (uma por referência)
- Tempo estimado: ~2-5 minutos

OTIMIZADO:
- Embeddings gerados: 100 + 100 = 200 (uma vez cada)
- Chamadas LLM: 10-20 (batch matching)
- Tempo estimado: ~10-30 segundos
```

---

## 3. PROPOSTA DE MELHORIAS PARA ETAPA 2

### 3.1 Arquitetura Proposta

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ETAPA 2: FLUXO OTIMIZADO (PROPOSTO)                      │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐
    │ REFERÊNCIA      │
    │ (da sessão)     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────────────────────────────────────┐
    │           NORMALIZAÇÃO PRÉ-PROCESSAMENTO            │
    │                                                      │
    │   ┌───────────────┐     ┌───────────────┐           │
    │   │ Normalizar    │     │ Normalizar    │           │
    │   │ Referências   │     │ Alvos         │           │
    │   │ (unidecode,   │     │ (unidecode,   │           │
    │   │  lowercase)   │     │  lowercase)   │           │
    │   └───────┬───────┘     └───────┬───────┘           │
    │           │                     │                   │
    │           └─────────┬───────────┘                   │
    │                     ▼                               │
    │           ┌───────────────┐                         │
    │           │ Mapeamento    │                         │
    │           │ Original→Norm │                         │
    │           └───────────────┘                         │
    └─────────────────────────────────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────────────────────────┐
    │         EMBEDDING BATCH (UMA VEZ)                    │
    │                                                      │
    │   ┌───────────────┐     ┌───────────────┐           │
    │   │ Embed REF     │     │ Embed TARGET  │           │
    │   │ (batch)       │     │ (batch)       │           │
    │   └───────┬───────┘     └───────┬───────┘           │
    │           │                     │                   │
    │           └─────────┬───────────┘                   │
    │                     ▼                               │
    │           ┌───────────────┐                         │
    │           │ Matriz de     │ → Cosine Similarity     │
    │           │ Similaridade  │ → Dimensão: N_ref × N_tgt │
    │           └───────────────┘                         │
    └─────────────────────────────────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────────────────────────┐
    │              MATCHING EM 3 FASES                     │
    │                                                      │
    │   FASE 1: EXACT MATCH                               │
    │   ┌─────────────────────────────────────────────┐   │
    │   │ • Comparação normalizada case-insensitive    │   │
    │   │ • Confiança: 1.0                            │   │
    │   │ • Remove dos próximos passos                │   │
    │   └─────────────────────────────────────────────┘   │
    │                                                      │
    │   FASE 2: HIGH-CONFIDENCE VECTOR MATCH              │
    │   ┌─────────────────────────────────────────────┐   │
    │   │ • Similaridade > 0.95                       │   │
    │   │ • Aceita automaticamente como "semantic"    │   │
    │   │ • Sem chamada LLM                           │   │
    │   └─────────────────────────────────────────────┘   │
    │                                                      │
    │   FASE 3: BATCH LLM VALIDATION                      │
    │   ┌─────────────────────────────────────────────┐   │
    │   │ • Agrupa candidatos (0.7 < sim < 0.95)      │   │
    │   │ • Batch de até 10 pares por chamada         │   │
    │   │ • Output JSON estruturado                   │   │
    │   └─────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────────────────────────┐
    │              VALIDAÇÃO CRUZADA                       │
    │                                                      │
    │   ┌─────────────────────────────────────────────┐   │
    │   │ • Verifica se A→B implica B→A               │   │
    │   │ • Detecta conflitos (A→B, A→C)              │   │
    │   │ • Resolve pelo maior score                  │   │
    │   └─────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────┘
             │
             ▼
    ┌─────────────────┐
    │  RESULTADOS     │ → Matches confirmados
    │  FINAIS         │ → Missing com explicação
    │                 │ → Extra com contexto
    └─────────────────┘
```

### 3.2 Melhorias Específicas

#### 3.2.1 Normalização Prévia
```python
def normalize_entity(entity: str) -> str:
    """Normaliza entidade para comparação."""
    from unidecode import unidecode

    normalized = entity.strip()
    normalized = unidecode(normalized)  # Remove acentos
    normalized = normalized.lower()
    normalized = re.sub(r'\s+', ' ', normalized)  # Normaliza espaços
    return normalized
```

#### 3.2.2 Matriz de Similaridade em Batch
```python
def compute_similarity_matrix(
    ref_embeddings: np.ndarray,
    target_embeddings: np.ndarray
) -> np.ndarray:
    """Calcula matriz de similaridade de coseno em batch."""
    # Normaliza os vetores
    ref_norm = ref_embeddings / np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
    target_norm = target_embeddings / np.linalg.norm(target_embeddings, axis=1, keepdims=True)

    # Produto matricial = matriz de similaridades
    return np.dot(ref_norm, target_norm.T)
```

#### 3.2.3 Matching em Três Fases
```python
def three_phase_matching(
    ref_entities: List[str],
    target_entities: List[str],
    similarity_matrix: np.ndarray
) -> List[EntityMatch]:
    """Matching otimizado em três fases."""

    # FASE 1: Exact matches (normalizado)
    exact_matches = find_exact_matches(ref_entities, target_entities)

    # FASE 2: High-confidence vector matches (>0.95)
    vector_matches = find_high_confidence_matches(
        similarity_matrix,
        threshold=0.95,
        exclude=exact_matches
    )

    # FASE 3: Batch LLM validation para candidatos intermediários
    llm_candidates = get_candidates_for_llm(
        similarity_matrix,
        min_threshold=0.7,
        max_threshold=0.95,
        exclude=exact_matches | vector_matches
    )
    llm_matches = batch_llm_validation(llm_candidates)

    return combine_results(exact_matches, vector_matches, llm_matches)
```

#### 3.2.4 Prompt de Batch Matching
```python
BATCH_MATCHING_PROMPT = """Você é um especialista em correspondência de entidades.

TAREFA: Analise os pares de entidades abaixo e determine se são correspondências válidas.

PARES PARA ANÁLISE:
{pairs_list}

INSTRUÇÕES:
1. Considere variações de: acentos, case, formatação, erros de digitação
2. Para nomes: "João Silva" = "Joao Silva" = "JOÃO SILVA" = "Silva, João"
3. Para números: "123-456" = "123456" = "123.456"

OUTPUT (JSON array):
[
  {{"pair_index": 0, "is_match": true, "confidence": 0.95, "type": "semantic", "reason": "..."}},
  {{"pair_index": 1, "is_match": false, "confidence": 0.2, "type": "no_match", "reason": "..."}}
]

RESPONDA APENAS COM O JSON:"""
```

### 3.3 Comparativo de Performance

| Métrica | Implementação Atual | Implementação Proposta | Melhoria |
|---------|---------------------|------------------------|----------|
| Embeddings por comparação | O(n×m) | O(n+m) | **~100x** |
| Chamadas LLM | n (uma por ref) | n/10 (batch) | **~10x** |
| Tempo (100×100) | ~2-5 min | ~10-30 seg | **~10x** |
| Acurácia (estimada) | 85-90% | 95-98% | **+5-10%** |
| Detecção de variações | Moderada | Alta | **Melhora** |

---

## 4. PRÓXIMOS PASSOS

### 4.1 Implementação Sugerida

1. **Fase 1 - Normalização** (Baixo risco)
   - Adicionar `unidecode` ao requirements.txt
   - Criar função de normalização
   - Aplicar antes do matching

2. **Fase 2 - Batch Embeddings** (Médio risco)
   - Refatorar `_get_vector_similarity_candidates`
   - Implementar matriz de similaridade
   - Cache de embeddings

3. **Fase 3 - Matching em 3 Fases** (Médio risco)
   - Implementar `three_phase_matching`
   - Criar prompt de batch
   - Validação cruzada

4. **Fase 4 - Testes e Validação** (Crítico)
   - Criar dataset de teste
   - Comparar métricas atual vs proposto
   - Ajustar thresholds

---

## 5. CONCLUSÃO

A **Etapa 1** é bem-sucedida devido a:
- Cobertura completa de documentos pequenos
- Prompt estruturado com exemplos
- Temperature 0.0 para determinismo
- Deduplicação inteligente

A **Etapa 2** pode ser significativamente melhorada aplicando os mesmos princípios:
- Pré-processamento robusto (normalização)
- Processamento em batch (embeddings + LLM)
- Thresholds adaptativos por fase
- Validação cruzada para consistência

As melhorias propostas devem reduzir o tempo de processamento em ~10x e aumentar a acurácia em ~5-10%.
