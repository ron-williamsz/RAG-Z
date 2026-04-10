"""API REST para o sistema RAG usando FastAPI."""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .embeddings import EmbeddingsManager
from .vector_store import VectorStore
from .rag_chain import RAGChain
from .context_manager import ContextManager
from .document_loader import DocumentLoader
from .chunker import Chunker
from .verification_engine import VerificationEngine
from .verification_models import (
    ExtractReferenceRequest,
    ExtractReferenceResponse,
    CompareTargetResponse,
)
from .governance import GovernanceManager, UserProfile
from .conversation_manager import ConversationManager, Message


# ==================== MODELS ====================

class ConversationMessage(BaseModel):
    """Mensagem de conversa."""
    role: str = Field(..., description="Papel: 'user' ou 'assistant'")
    content: str = Field(..., description="Conteúdo da mensagem")


class QueryRequest(BaseModel):
    """Modelo de requisição para consultas."""
    question: str = Field(..., description="Pergunta do usuário", min_length=1)
    context: str = Field("default", description="Nome do contexto a usar")
    llm_provider: str = Field("openai", description="Provedor do LLM (openai, anthropic)")
    embedding_provider: str = Field("ollama", description="Provedor de embeddings (ollama, openai)")
    top_k: int = Field(8, description="Número de chunks a recuperar", ge=1, le=20)
    return_sources: bool = Field(True, description="Se deve retornar fontes no JSON")
    use_legal_hierarchy: bool = Field(False, description="Usar hierarquia legal (codigo_civil > lei_condominios > contexto)")

    # Modo de resposta
    show_source: bool = Field(False, description="Se True, menciona a fonte NA RESPOSTA (ex: 'De acordo com a convenção...')")
    fluent_mode: bool = Field(True, description="Se True, resposta fluida sem mencionar hierarquia técnica")

    # Histórico de conversa
    conversation_history: Optional[List[ConversationMessage]] = Field(None, description="Histórico de mensagens da conversa")
    conversation_id: Optional[str] = Field(None, description="ID da conversa no chatbot externo")

    # Autenticação e perfil
    user_id: Optional[str] = Field(None, description="ID do usuário")
    is_authenticated: bool = Field(False, description="Se o usuário está autenticado")
    is_admin: bool = Field(False, description="Se o usuário é administrador")


class HierarchicalQueryRequest(BaseModel):
    """Modelo de requisição para consultas com hierarquia legal completa."""
    question: str = Field(..., description="Pergunta do usuário", min_length=1)
    context: str = Field("zangari_website", description="Contexto principal (ex: cond_0388, zangari_website)")
    llm_provider: str = Field("openai", description="Provedor do LLM (openai, anthropic)")
    embedding_provider: str = Field("ollama", description="Provedor de embeddings (ollama, openai)")
    top_k_per_context: int = Field(3, description="Chunks por contexto hierárquico", ge=1, le=10)
    return_sources: bool = Field(True, description="Se deve retornar fontes no JSON")

    # Modo de resposta
    show_source: bool = Field(False, description="Se True, menciona a fonte NA RESPOSTA (ex: 'De acordo com a convenção...')")
    fluent_mode: bool = Field(True, description="Se True, resposta fluida sem mencionar hierarquia técnica")

    # Hierarquia específica
    hierarchy_level: Optional[str] = Field(None, description="Nível específico: convencao, regimento_interno, etc")
    strict_hierarchy: bool = Field(False, description="Se True, só retorna do nível pedido")

    # Histórico de conversa
    conversation_history: Optional[List[ConversationMessage]] = Field(None, description="Histórico de mensagens")
    conversation_id: Optional[str] = Field(None, description="ID da conversa no chatbot externo")

    # Autenticação e perfil
    user_id: Optional[str] = Field(None, description="ID do usuário")
    is_authenticated: bool = Field(False, description="Se o usuário está autenticado")
    is_admin: bool = Field(False, description="Se o usuário é administrador")


class HierarchicalQueryResponse(BaseModel):
    """Modelo de resposta para consultas hierárquicas."""
    answer: str
    sources: Optional[List[dict]] = None
    contexts_searched: List[str] = Field(default_factory=list, description="Todos os contextos pesquisados")
    contexts_with_results: Optional[List[str]] = Field(default=None, description="Contextos onde encontrou resultados")
    hierarchy_applied: bool
    hierarchy_level: Optional[str] = None
    found_in_requested_level: Optional[bool] = Field(default=None, description="Se encontrou no nível solicitado")
    fallback_used: Optional[bool] = Field(default=None, description="Se usou fallback de outros níveis")
    user_profile: str = "anonymous"
    llm_provider: str
    embedding_provider: str
    context_format: Optional[str] = None
    conversation_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Modelo de resposta para consultas."""
    answer: str
    sources: Optional[List[dict]] = None
    context: str
    llm_provider: str
    embedding_provider: str
    context_format: Optional[str] = None


class IndexResponse(BaseModel):
    """Modelo de resposta para indexação."""
    status: str
    message: str
    files_indexed: List[str]
    total_chunks: int
    context: str
    embedding_provider: str


class ContextInfo(BaseModel):
    """Informações sobre um contexto."""
    name: str
    total_files: int
    total_chunks: int
    created_at: Optional[str] = None
    last_updated: Optional[str] = None


class ContextListResponse(BaseModel):
    """Lista de contextos disponíveis."""
    contexts: List[ContextInfo]
    total: int


class ContextCreateRequest(BaseModel):
    """Requisição para criar novo contexto."""
    name: str = Field(..., description="Nome do contexto", min_length=1)


class ContextCreateResponse(BaseModel):
    """Resposta da criação de contexto."""
    status: str
    message: str
    context_name: str


class HealthResponse(BaseModel):
    """Resposta do health check."""
    status: str
    version: str
    available_contexts: int
    embeddings_provider: str


# ==================== API KEY AUTH ====================

RAG_API_KEY = os.getenv("RAG_API_KEY", "")

# Endpoints públicos (sem API key)
PUBLIC_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json", "/api/query", "/api/query/hierarchical"}


def verify_api_key(request: Request, x_api_key: str = Header(None)):
    """Verifica API key para endpoints protegidos."""
    if not RAG_API_KEY:
        return  # Se não há key configurada, tudo aberto (dev mode)
    if request.url.path in PUBLIC_PATHS:
        return
    if request.method == "GET" and request.url.path.startswith("/api/contexts"):
        return  # GET de contextos é público (listagem)
    if x_api_key == RAG_API_KEY:
        return
    raise HTTPException(status_code=403, detail="API key inválida ou ausente. Envie header X-API-Key.")


# ==================== APP ====================

app = FastAPI(
    title="RAG API",
    description="API REST para sistema RAG (Retrieval-Augmented Generation)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    dependencies=[Depends(verify_api_key)],
)

# CORS - Permitir requisições de qualquer origem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global verification engine instance
verification_engine = VerificationEngine()

# ==================== CONFIG PERSISTENTE ====================

RAG_CONFIG_PATH = Path("data/rag_config.json")


def _load_rag_config() -> Dict[str, Any]:
    """Carrega configuração persistente do RAG."""
    if RAG_CONFIG_PATH.exists():
        try:
            with open(RAG_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_rag_config(config: Dict[str, Any]) -> None:
    """Salva configuração persistente do RAG."""
    RAG_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RAG_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# Contextos públicos (acessíveis para anônimos)
PUBLIC_CONTEXTS = ["zangari_website", "zangari_faq", "codigo_civil", "lei_condominios"]

DEFAULT_ANONYMOUS_PROMPT = """Você é o assistente virtual público do Grupo Zangari.

CONTEXTO ATIVO: {context_label} ({context_name})

CONTEXTOS PÚBLICOS DISPONÍVEIS:
{available_contexts}

DIRETRIZES:
- Responda APENAS com base nos documentos indexados no contexto ativo
- NÃO invente informações que não estejam nos documentos
- Se a informação não estiver disponível, informe que não encontrou nos documentos
- NÃO solicite dados pessoais do usuário
- Se o assunto exigir documentos restritos (convenção, regimento de condomínio específico), informe que esse conteúdo está disponível apenas para usuários autenticados
- Seja cordial, claro e objetivo

"""


def get_anonymous_prompt(
    context_name: str,
    available_public_contexts: Optional[List[str]] = None,
) -> str:
    """
    Gera prompt de sistema para usuários anônimos.

    Usa prompt customizado do rag_config.json se existir,
    caso contrário usa o DEFAULT_ANONYMOUS_PROMPT.

    Args:
        context_name: Nome técnico do contexto ativo (ex: zangari_website)
        available_public_contexts: Lista de contextos públicos disponíveis

    Returns:
        Prompt formatado para prepend à mensagem do usuário
    """
    # Verifica se há prompt customizado
    config = _load_rag_config()
    prompt_template = config.get("anonymous_prompt", "") or DEFAULT_ANONYMOUS_PROMPT

    # Monta label amigável
    context_label = HIERARCHY_NAMES.get(context_name, context_name.replace("_", " ").title())

    # Lista de contextos públicos disponíveis
    if available_public_contexts is None:
        cm = _get_context_manager()
        all_contexts = cm.list_contexts()
        available_public_contexts = [c for c in PUBLIC_CONTEXTS if c in all_contexts]

    contexts_text = "\n".join([
        f"- {HIERARCHY_NAMES.get(c, c.replace('_', ' ').title())} ({c})"
        for c in available_public_contexts
    ]) if available_public_contexts else "- Nenhum contexto público disponível"

    return prompt_template.format(
        context_name=context_name,
        context_label=context_label,
        available_contexts=contexts_text,
    )


# ==================== HELPERS ====================

# Hierarquia legal para busca em múltiplos contextos
# Ordem: 1º Código Civil, 2º Lei de Condomínios, 3º Contexto do Condomínio
LEGAL_HIERARCHY_CONTEXTS = ["codigo_civil", "lei_condominios"]

# Mapeamento de nomes amigáveis para níveis hierárquicos
HIERARCHY_NAMES = {
    "codigo_civil": "Código Civil (Lei 10.406/2002)",
    "lei_condominios": "Lei de Condomínios (Lei 4.591/64)",
    "convencao": "Convenção do Condomínio",
    "regimento_interno": "Regimento Interno",
    "ata_assembleia": "Ata de Assembleia",
    "avisos": "Avisos e Comunicados",
}

# Níveis hierárquicos (menor = maior prioridade)
HIERARCHY_LEVELS = {
    "codigo_civil": 1,
    "lei_condominios": 2,
    "convencao": 3,
    "regimento_interno": 4,
    "ata_assembleia": 5,
    "avisos": 6,
}


def _get_hierarchy_metadata(
    context: str,
    hierarchy_level: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Determina metadados de hierarquia baseado no contexto e nível.

    Args:
        context: Nome do contexto (ex: codigo_civil, cond_0388)
        hierarchy_level: Nível hierárquico explícito (opcional)

    Returns:
        Dict com metadados de hierarquia
    """
    # Se é um contexto de legislação federal
    if context in LEGAL_HIERARCHY_CONTEXTS:
        return {
            "hierarchy_context": context,
            "hierarchy_name": HIERARCHY_NAMES.get(context, context),
            "hierarchy_level": HIERARCHY_LEVELS.get(context, 99),
            "is_legal_source": True,
        }

    # Se foi especificado um nível hierárquico
    if hierarchy_level:
        level_key = hierarchy_level.lower().replace(" ", "_")
        return {
            "hierarchy_context": context,
            "hierarchy_name": HIERARCHY_NAMES.get(level_key, hierarchy_level),
            "hierarchy_level": HIERARCHY_LEVELS.get(level_key, 99),
            "is_legal_source": False,
        }

    # Contexto de condomínio sem nível especificado
    return {
        "hierarchy_context": context,
        "hierarchy_name": f"Documentos do {context}",
        "hierarchy_level": 99,  # Menor prioridade
        "is_legal_source": False,
    }


def _get_context_manager() -> ContextManager:
    """Retorna instância do ContextManager."""
    return ContextManager()


def _search_with_cascade(
    question: str,
    condo_context: str,
    embedding_provider: str,
    top_k_per_context: int = 3,
    score_threshold: float = 0.7,
    requested_level: Optional[str] = None,
    strict_hierarchy: bool = False,
) -> Dict[str, Any]:
    """
    Busca em CASCATA respeitando hierarquia legal.

    Comportamento:
    1. Se pediu nível específico (ex: convenção), busca PRIMEIRO nele
    2. Se não encontrou (ou não pediu específico), busca na hierarquia em ordem
    3. Retorna APENAS os contextos onde ENCONTROU algo relevante
    4. Não menciona contextos onde não encontrou nada

    Hierarquia padrão para condomínio:
    1º Convenção → 2º Regimento → 3º Lei de Condomínios → 4º Código Civil → 5º Atas

    Args:
        question: Pergunta do usuário
        condo_context: Contexto do condomínio (ex: cond_0388)
        embedding_provider: Provedor de embeddings
        top_k_per_context: Número de chunks por contexto
        score_threshold: Score mínimo para considerar relevante
        requested_level: Nível específico solicitado (ex: "convencao")
        strict_hierarchy: Se True e pediu nível específico, só retorna desse nível

    Returns:
        Dict com:
        - documents: Lista de documentos encontrados
        - contexts_with_results: Contextos onde encontrou algo
        - contexts_searched: Todos os contextos pesquisados
        - found_in_requested: Se encontrou no nível solicitado
        - fallback_used: Se usou fallback de outros níveis
    """
    cm = _get_context_manager()
    available_contexts = cm.list_contexts()

    # Hierarquia para condomínio (do mais específico para o mais geral)
    condo_hierarchy = [
        condo_context,  # Documentos do condomínio (convenção, regimento, etc)
        "lei_condominios",
        "codigo_civil",
    ]

    # Se pediu nível específico, reorganiza para buscar primeiro nele
    if requested_level:
        level_key = requested_level.lower().replace(" ", "_")
        # Se é um dos níveis de legislação
        if level_key in ["codigo_civil", "lei_condominios"]:
            condo_hierarchy = [level_key] + [c for c in condo_hierarchy if c != level_key]

    # Cria embeddings manager uma vez
    embeddings_manager = EmbeddingsManager.from_config(
        config_path="config.toml",
        override_provider=embedding_provider,
    )

    all_documents = []
    contexts_with_results = []
    contexts_searched = []
    found_in_requested = False

    for ctx in condo_hierarchy:
        if ctx not in available_contexts:
            continue

        contexts_searched.append(ctx)

        try:
            # Cria vector store para este contexto
            vector_store = VectorStore(
                embeddings=embeddings_manager.embeddings,
                context_name=ctx,
            )
            vector_store.load()

            # Busca documentos com scores
            results = vector_store.search(
                query=question,
                top_k=top_k_per_context,
            )

            # Filtra por score threshold - só pega documentos relevantes
            relevant_docs = []
            for doc, score in results:
                # FAISS retorna distância (menor = melhor), convertemos para similaridade
                similarity = 1 - score if score <= 1 else 1 / (1 + score)
                if similarity >= score_threshold:
                    # Adiciona metadados de hierarquia
                    doc.metadata["hierarchy_level"] = condo_hierarchy.index(ctx) + 1
                    doc.metadata["hierarchy_context"] = ctx
                    doc.metadata["hierarchy_name"] = HIERARCHY_NAMES.get(ctx, f"Documentos do {ctx}")
                    doc.metadata["relevance_score"] = round(similarity, 3)
                    relevant_docs.append(doc)

            if relevant_docs:
                all_documents.extend(relevant_docs)
                contexts_with_results.append(ctx)

                # Verifica se encontrou no nível solicitado
                if requested_level:
                    level_key = requested_level.lower().replace(" ", "_")
                    if ctx == level_key or level_key in ctx:
                        found_in_requested = True

        except FileNotFoundError:
            continue
        except Exception:
            continue

    # Se strict_hierarchy e pediu nível específico, filtra apenas esse nível
    if strict_hierarchy and requested_level and found_in_requested:
        level_key = requested_level.lower().replace(" ", "_")
        all_documents = [
            doc for doc in all_documents
            if doc.metadata.get("hierarchy_context") == level_key
            or level_key in doc.metadata.get("hierarchy_context", "")
        ]
        contexts_with_results = [ctx for ctx in contexts_with_results if level_key in ctx]

    return {
        "documents": all_documents,
        "contexts_with_results": contexts_with_results,
        "contexts_searched": contexts_searched,
        "found_in_requested": found_in_requested,
        "fallback_used": bool(contexts_with_results) and not found_in_requested if requested_level else False,
        "requested_level": requested_level,
    }


def _search_with_hierarchy(
    question: str,
    condo_context: str,
    embedding_provider: str,
    top_k_per_context: int = 3,
) -> tuple[list, list[str]]:
    """
    Busca em múltiplos contextos respeitando hierarquia legal.
    (Mantido para compatibilidade - usa nova função internamente)

    Returns:
        Tuple de (documentos combinados, contextos encontrados)
    """
    result = _search_with_cascade(
        question=question,
        condo_context=condo_context,
        embedding_provider=embedding_provider,
        top_k_per_context=top_k_per_context,
    )
    return result["documents"], result["contexts_with_results"]


def _build_rag_chain(
    context_name: str,
    llm_provider: str,
    embedding_provider: str,
    top_k: int,
) -> RAGChain:
    """
    Constrói RAGChain para um contexto específico.

    Args:
        context_name: Nome do contexto
        llm_provider: Provedor do LLM (openai, anthropic)
        embedding_provider: Provedor de embeddings (ollama, openai)
        top_k: Número de chunks a recuperar

    Returns:
        Instância configurada do RAGChain

    Raises:
        HTTPException: Se contexto não existir ou houver erro
    """
    try:
        # Verifica se contexto existe
        cm = _get_context_manager()
        if context_name not in cm.list_contexts():
            raise HTTPException(
                status_code=404,
                detail=f"Contexto '{context_name}' não encontrado. Use /api/contexts para ver contextos disponíveis."
            )

        # Cria embeddings manager
        embeddings_manager = EmbeddingsManager.from_config(
            config_path="config.toml",
            override_provider=embedding_provider,
        )

        # Cria vector store e carrega índice
        vector_store = VectorStore(
            embeddings=embeddings_manager.embeddings,
            context_name=context_name,
        )

        try:
            vector_store.load()
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Índice FAISS não encontrado para contexto '{context_name}'. Indexe documentos primeiro usando /api/index."
            )

        # Cria RAG chain
        rag_chain = RAGChain.from_config(
            vector_store=vector_store,
            config_path="config.toml",
            llm_provider=llm_provider,
            context_name=context_name,
        )
        rag_chain.top_k = top_k

        return rag_chain

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao construir RAG chain: {str(e)}"
        )


# ==================== ENDPOINTS ====================

@app.get("/", tags=["Root"])
def root():
    """Endpoint raiz - informações básicas da API."""
    return {
        "name": "RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check - verifica se a API está funcionando."""
    try:
        cm = _get_context_manager()
        contexts = cm.list_contexts()

        return {
            "status": "healthy",
            "version": "1.0.0",
            "available_contexts": len(contexts),
            "embeddings_provider": "ollama",  # Pode ler do config.toml
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
def query(req: QueryRequest):
    """
    Executa uma consulta no sistema RAG.

    - **question**: Pergunta do usuário
    - **context**: Nome do contexto (padrão: "default")
    - **llm_provider**: LLM a usar (openai, anthropic)
    - **embedding_provider**: Provedor de embeddings (ollama, openai)
    - **top_k**: Número de chunks a recuperar (padrão: 8)
    - **return_sources**: Se deve retornar fontes (padrão: true)
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Pergunta vazia.")

    # Valida provedores
    if req.llm_provider not in ["openai", "anthropic"]:
        raise HTTPException(
            status_code=400,
            detail=f"LLM provider inválido: '{req.llm_provider}'. Use 'openai' ou 'anthropic'."
        )

    if req.embedding_provider not in ["ollama", "openai"]:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding provider inválido: '{req.embedding_provider}'. Use 'ollama' ou 'openai'."
        )

    try:
        # Constrói RAG chain
        rag_chain = _build_rag_chain(
            context_name=req.context,
            llm_provider=req.llm_provider,
            embedding_provider=req.embedding_provider,
            top_k=req.top_k,
        )

        # Executa query
        result = rag_chain.query(req.question, return_sources=req.return_sources)

        return {
            "answer": result["answer"],
            "sources": result.get("sources") if req.return_sources else None,
            "context": req.context,
            "llm_provider": req.llm_provider,
            "embedding_provider": req.embedding_provider,
            "context_format": result.get("context_format"),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar query: {str(e)}"
        )


@app.post("/api/query/hierarchical", response_model=HierarchicalQueryResponse, tags=["Query"])
def query_hierarchical(req: HierarchicalQueryRequest):
    """
    Executa consulta com hierarquia legal, histórico de conversa e perfis de acesso.

    Busca automaticamente em múltiplos contextos respeitando a hierarquia:
    1️⃣ **Código Civil** (Lei 10.406/2002) - Norma suprema
    2️⃣ **Lei de Condomínios** (Lei 4.591/64) - Segunda na hierarquia
    3️⃣ **Documentos do Condomínio** (Convenção, Regimento, Atas)

    Recursos:
    - **Hierarquia Legal**: Busca em ordem de precedência
    - **Histórico de Conversa**: Contexto de mensagens anteriores
    - **Perfis de Acesso**: Controle de quais documentos o usuário pode ver
    - **Nível Específico**: Pode solicitar resposta de um nível específico (ex: só convenção)
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Pergunta vazia.")

    # Valida provedores
    if req.llm_provider not in ["openai", "anthropic"]:
        raise HTTPException(
            status_code=400,
            detail=f"LLM provider inválido: '{req.llm_provider}'. Use 'openai' ou 'anthropic'."
        )

    if req.embedding_provider not in ["ollama", "openai"]:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding provider inválido: '{req.embedding_provider}'. Use 'ollama' ou 'openai'."
        )

    try:
        # 1. Inicializa GovernanceManager
        governance = GovernanceManager(config_path="config.toml")

        # 2. Determina perfil do usuário
        user_profile = governance.get_user_profile(
            user_id=req.user_id,
            is_authenticated=req.is_authenticated,
            is_admin=req.is_admin,
        )

        # 3. Valida contextos permitidos
        allowed_contexts = governance.get_allowed_contexts(
            requested_context=req.context,
            profile=user_profile,
        )

        if not allowed_contexts:
            raise HTTPException(
                status_code=403,
                detail=f"Acesso negado ao contexto '{req.context}' para perfil '{user_profile.value}'."
            )

        # 4. Se anônimo, prepara prompt anônimo para contextualizar
        is_anonymous = user_profile == UserProfile.ANONYMOUS
        anonymous_context = ""
        if is_anonymous:
            anonymous_context = get_anonymous_prompt(
                context_name=req.context,
            )

        # 5. Processa histórico de conversa
        conversation_history_text = ""
        if req.conversation_history:
            # Usa histórico do request
            history_parts = []
            for msg in req.conversation_history:
                role_label = "USUÁRIO" if msg.role == "user" else "ASSISTENTE"
                history_parts.append(f"**{role_label}**: {msg.content}")
            conversation_history_text = "\n".join(history_parts)

        elif req.conversation_id:
            # Busca histórico do chatbot externo
            conv_manager = ConversationManager.from_governance(governance)
            messages = conv_manager.fetch_conversation_history_sync(req.conversation_id)
            if messages:
                conversation_history_text = conv_manager.format_history_for_prompt(messages)

        # 5. Busca em CASCATA respeitando hierarquia
        search_result = _search_with_cascade(
            question=req.question,
            condo_context=req.context,
            embedding_provider=req.embedding_provider,
            top_k_per_context=req.top_k_per_context,
            requested_level=req.hierarchy_level,
            strict_hierarchy=req.strict_hierarchy,
        )

        documents = search_result["documents"]
        contexts_with_results = search_result["contexts_with_results"]
        found_in_requested = search_result["found_in_requested"]
        fallback_used = search_result["fallback_used"]

        # 6. Monta mensagem de not_found apropriada
        if not documents:
            # Não encontrou NADA em nenhum nível da hierarquia
            not_found_msg = "Não encontrei informações sobre este assunto em nenhum dos documentos disponíveis."
            if req.hierarchy_level:
                level_name = governance.get_hierarchy_level_name(req.hierarchy_level)
                not_found_msg = f"Não encontrei informações sobre este assunto na {level_name}, nem em outros documentos da hierarquia."

            return {
                "answer": not_found_msg,
                "sources": [],
                "contexts_searched": search_result["contexts_searched"],
                "contexts_with_results": [],
                "hierarchy_applied": True,
                "hierarchy_level": req.hierarchy_level,
                "found_in_requested_level": False,
                "fallback_used": False,
                "user_profile": user_profile.value,
                "llm_provider": req.llm_provider,
                "embedding_provider": req.embedding_provider,
                "context_format": "hierarchical",
                "conversation_id": req.conversation_id,
            }

        # 7. Prepara mensagem de fallback se necessário
        fallback_notice = ""
        if req.hierarchy_level and fallback_used and governance.should_include_fallback_notice():
            level_name = governance.get_hierarchy_level_name(req.hierarchy_level)
            found_in_names = [HIERARCHY_NAMES.get(ctx, ctx) for ctx in contexts_with_results]
            fallback_notice = f"Não encontrei informações específicas na {level_name}, mas encontrei conteúdo relevante em: {', '.join(found_in_names)}."

        # 8. Cria RAGChain e gera resposta
        embeddings_manager = EmbeddingsManager.from_config(
            config_path="config.toml",
            override_provider=req.embedding_provider,
        )

        vector_store = VectorStore(
            embeddings=embeddings_manager.embeddings,
            context_name="temp_hierarchical",
        )

        rag_chain = RAGChain.from_config(
            vector_store=vector_store,
            config_path="config.toml",
            llm_provider=req.llm_provider,
            context_name=req.context,
        )

        # 9. Combina contexto anônimo + histórico para o LLM
        combined_history = ""
        if anonymous_context:
            combined_history = anonymous_context
        if conversation_history_text:
            combined_history = f"{combined_history}\n{conversation_history_text}" if combined_history else conversation_history_text

        # 10. Gera resposta baseado no modo (fluent ou técnico)
        not_found_msg = governance.get_not_found_message(req.hierarchy_level)

        if req.fluent_mode:
            # Modo fluido: resposta amigável sem jargões técnicos
            result = rag_chain.query_fluent(
                question=req.question,
                documents=documents,
                conversation_history=combined_history if combined_history else None,
                show_source=req.show_source,
                not_found_message=not_found_msg,
                return_sources=req.return_sources,
            )
        else:
            # Modo técnico: menciona hierarquia legal
            result = rag_chain.query_with_history(
                question=req.question,
                documents=documents,
                conversation_history=combined_history if combined_history else None,
                not_found_message=not_found_msg,
                return_sources=req.return_sources,
            )

        # 11. Adiciona aviso de fallback se necessário
        answer = result["answer"]
        if fallback_notice and not req.fluent_mode:
            answer = f"{fallback_notice}\n\n{answer}"

        # 12. Prepara resposta final
        return {
            "answer": answer,
            "sources": result.get("sources") if req.return_sources else None,
            "contexts_searched": search_result["contexts_searched"],
            "contexts_with_results": contexts_with_results,
            "hierarchy_applied": True,
            "hierarchy_level": req.hierarchy_level,
            "found_in_requested_level": found_in_requested,
            "fallback_used": fallback_used,
            "user_profile": user_profile.value,
            "llm_provider": req.llm_provider,
            "embedding_provider": req.embedding_provider,
            "context_format": result.get("context_format", "hierarchical"),
            "conversation_id": req.conversation_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar query hierárquica: {str(e)}"
        )


@app.post("/api/index", response_model=IndexResponse, tags=["Indexing"])
async def index_documents(
    files: List[UploadFile] = File(..., description="Arquivos para indexar"),
    context: str = Form("default", description="Nome do contexto"),
    embedding_provider: str = Form("ollama", description="Provedor de embeddings (ollama, openai)"),
    hierarchy_level: str = Form(None, description="Nível hierárquico: convencao, regimento_interno, ata_assembleia, avisos"),
):
    """
    Indexa documentos em um contexto específico.

    - **files**: Arquivos para indexar (PDF, DOCX, XLSX, TXT, MD)
    - **context**: Nome do contexto (padrão: "default")
    - **embedding_provider**: Provedor de embeddings (ollama, openai)
    - **hierarchy_level**: Nível na hierarquia legal (opcional): convencao, regimento_interno, ata_assembleia, avisos
    """
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo fornecido.")

    # Valida provider
    if embedding_provider not in ["ollama", "openai"]:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding provider inválido: '{embedding_provider}'. Use 'ollama' ou 'openai'."
        )

    try:
        # Cria contexto se não existir
        cm = _get_context_manager()
        if context not in cm.list_contexts():
            cm.create_context(context)

        # Cria diretório temporário para upload
        upload_dir = Path("data/temp_uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Salva arquivos temporariamente
        saved_files = []
        for file in files:
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append(str(file_path))

        # Carrega documentos
        loader = DocumentLoader()
        all_documents = []
        file_names = []

        # Determina metadados de hierarquia baseado no contexto
        hierarchy_metadata = _get_hierarchy_metadata(context, hierarchy_level)

        for file_path in saved_files:
            docs = loader.load(file_path)

            # Adiciona metadados de hierarquia a cada documento
            for doc in docs:
                doc.metadata.update(hierarchy_metadata)

            all_documents.extend(docs)
            file_names.append(Path(file_path).name)

        # Chunking
        chunker = Chunker.from_config("config.toml")
        chunks = chunker.split(all_documents)

        # Cria embeddings manager
        embeddings_manager = EmbeddingsManager.from_config(
            config_path="config.toml",
            override_provider=embedding_provider,
        )

        # Cria/carrega vector store
        vector_store = VectorStore(
            embeddings=embeddings_manager.embeddings,
            context_name=context,
        )

        # Tenta carregar índice existente
        try:
            vector_store.load()
        except FileNotFoundError:
            # Índice não existe, será criado ao adicionar documentos
            pass

        # Adiciona documentos
        vector_store.add_documents(chunks)

        # Salva índice
        vector_store.save(file_names=file_names)

        # Atualiza metadados do contexto
        stats = vector_store.get_stats()
        cm.update_context_metadata(context, file_names, stats.get("total_documents", 0))

        # Limpa arquivos temporários
        for file_path in saved_files:
            os.remove(file_path)

        return {
            "status": "success",
            "message": f"Documentos indexados com sucesso no contexto '{context}'",
            "files_indexed": file_names,
            "total_chunks": len(chunks),
            "context": context,
            "embedding_provider": embedding_provider,
        }

    except Exception as e:
        # Limpa arquivos temporários em caso de erro
        for file_path in saved_files:
            if os.path.exists(file_path):
                os.remove(file_path)

        raise HTTPException(
            status_code=500,
            detail=f"Erro ao indexar documentos: {str(e)}"
        )


@app.get("/api/contexts", response_model=ContextListResponse, tags=["Contexts"])
def list_contexts():
    """
    Lista todos os contextos disponíveis.

    Retorna informações sobre cada contexto incluindo:
    - Nome
    - Número de arquivos indexados
    - Número de chunks
    - Data de criação/atualização
    """
    try:
        cm = _get_context_manager()
        contexts = cm.list_contexts()

        context_infos = []
        for context_name in contexts:
            try:
                metadata = cm.get_context_metadata(context_name) or {}
                context_infos.append({
                    "name": context_name,
                    "total_documents": metadata.get("total_documents", 0),
                    "total_files": len(metadata.get("indexed_files", [])),
                    "total_chunks": metadata.get("total_documents", 0),
                    "created_at": metadata.get("created_at"),
                    "last_updated": metadata.get("last_updated"),
                })
            except Exception:
                context_infos.append({
                    "name": context_name,
                    "total_files": 0,
                    "total_chunks": 0,
                    "created_at": None,
                    "last_updated": None,
                })

        return {
            "contexts": context_infos,
            "total": len(context_infos),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao listar contextos: {str(e)}"
        )


@app.post("/api/contexts", response_model=ContextCreateResponse, tags=["Contexts"])
def create_context(req: ContextCreateRequest):
    """
    Cria um novo contexto vazio.

    - **name**: Nome do contexto (deve ser único)
    """
    try:
        cm = _get_context_manager()

        # Verifica se já existe
        if req.name in cm.list_contexts():
            raise HTTPException(
                status_code=409,
                detail=f"Contexto '{req.name}' já existe."
            )

        # Cria contexto
        cm.create_context(req.name)

        return {
            "status": "success",
            "message": f"Contexto '{req.name}' criado com sucesso",
            "context_name": req.name,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao criar contexto: {str(e)}"
        )


@app.delete("/api/contexts/{context_name}", tags=["Contexts"])
def delete_context(context_name: str):
    """
    Deleta um contexto e todos os seus dados.

    ⚠️ **ATENÇÃO**: Esta operação é irreversível!

    - **context_name**: Nome do contexto a deletar
    """
    try:
        cm = _get_context_manager()

        # Verifica se existe
        if context_name not in cm.list_contexts():
            raise HTTPException(
                status_code=404,
                detail=f"Contexto '{context_name}' não encontrado."
            )

        # Não permite deletar o contexto default
        if context_name == "default":
            raise HTTPException(
                status_code=403,
                detail="Não é permitido deletar o contexto 'default'."
            )

        # Deleta contexto
        cm.delete_context(context_name)

        return {
            "status": "success",
            "message": f"Contexto '{context_name}' deletado com sucesso",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao deletar contexto: {str(e)}"
        )


@app.get("/api/contexts/{context_name}/stats", tags=["Contexts"])
def get_context_stats(context_name: str):
    """
    Obtém estatísticas detalhadas de um contexto.

    - **context_name**: Nome do contexto
    """
    try:
        cm = _get_context_manager()

        # Verifica se existe
        if context_name not in cm.list_contexts():
            raise HTTPException(
                status_code=404,
                detail=f"Contexto '{context_name}' não encontrado."
            )

        # Obtém metadados do contexto
        metadata = cm.get_context_metadata(context_name) or {}

        return {
            "context_name": context_name,
            "total_documents": metadata.get("total_documents", 0),
            "total_files": len(metadata.get("indexed_files", [])),
            "indexed_files": metadata.get("indexed_files", []),
            "last_updated": metadata.get("last_updated"),
            "created_at": metadata.get("created_at"),
            "description": metadata.get("description", ""),
            "has_index": cm.has_index(context_name),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao obter estatísticas: {str(e)}"
        )


# ==================== EXPORT / IMPORT ENDPOINTS ====================


class ImportRequest(BaseModel):
    """Modelo de requisição para importação de contexto."""
    context_name: str = Field(..., description="Nome do contexto a criar/atualizar")
    description: str = Field("", description="Descrição do contexto")
    documents: List[Dict[str, Any]] = Field(..., description="Lista de documentos (page_content + metadata)")
    embedding_provider: str = Field("ollama", description="Provedor de embeddings para re-indexar")


@app.get("/api/contexts/{context_name}/export", tags=["Export/Import"])
def export_context(context_name: str):
    """
    Exporta todos os documentos (texto + metadata) de um contexto como JSON.

    Use para migrar dados entre máquinas ou ambientes (local → nuvem).
    O JSON exportado é independente de modelo de embedding e pode ser
    re-indexado em qualquer instância com POST /api/contexts/{nome}/import.

    - **context_name**: Nome do contexto a exportar
    """
    try:
        cm = _get_context_manager()

        if context_name not in cm.list_contexts():
            raise HTTPException(
                status_code=404,
                detail=f"Contexto '{context_name}' não encontrado."
            )

        if not cm.has_index(context_name):
            raise HTTPException(
                status_code=404,
                detail=f"Contexto '{context_name}' não possui índice FAISS."
            )

        # Carrega o índice com embeddings dummy apenas para ler os documentos
        # Precisamos de qualquer embedding para instanciar o FAISS, mas não vamos
        # gerar novos embeddings - apenas ler o docstore
        embeddings_manager = EmbeddingsManager.from_config(config_path="config.toml")
        vector_store = VectorStore(
            embeddings=embeddings_manager.embeddings,
            context_name=context_name,
        )
        vector_store.load()

        # Extrai todos os documentos do docstore do FAISS
        docstore = vector_store._vectorstore.docstore._dict
        documents = []
        for doc_id, doc in docstore.items():
            documents.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            })

        # Metadados do contexto
        context_metadata = cm.get_context_metadata(context_name)

        return {
            "status": "success",
            "context_name": context_name,
            "context_metadata": context_metadata,
            "total_documents": len(documents),
            "documents": documents,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao exportar contexto: {str(e)}"
        )


@app.post("/api/contexts/{context_name}/import", tags=["Export/Import"])
def import_context(context_name: str, req: ImportRequest):
    """
    Importa documentos e re-indexa em um contexto.

    Recebe o JSON exportado por GET /api/contexts/{nome}/export e
    re-indexa os documentos com o modelo de embedding da máquina local.

    - **context_name**: Nome do contexto destino
    - **documents**: Lista de documentos (page_content + metadata)
    - **embedding_provider**: Provedor de embeddings (ollama, openai)
    """
    try:
        from langchain_core.documents import Document

        cm = _get_context_manager()

        # Cria contexto se não existir
        if context_name not in cm.list_contexts():
            cm.create_context(context_name, req.description)

        if not req.documents:
            raise HTTPException(
                status_code=400,
                detail="Lista de documentos vazia."
            )

        # Converte dicts para Documents do LangChain
        documents = []
        for doc_data in req.documents:
            documents.append(
                Document(
                    page_content=doc_data["page_content"],
                    metadata=doc_data.get("metadata", {}),
                )
            )

        # Inicializa embeddings da máquina local
        embeddings_manager = EmbeddingsManager.from_config(
            config_path="config.toml",
            override_provider=req.embedding_provider,
        )

        # Cria o VectorStore e indexa
        vector_store = VectorStore(
            embeddings=embeddings_manager.embeddings,
            context_name=context_name,
        )

        # Tenta carregar índice existente para merge
        try:
            vector_store.load()
            vector_store.add_documents(documents)
        except FileNotFoundError:
            vector_store.create_index(documents)

        # Extrai nomes dos arquivos das metadata dos documentos
        file_names = list(set(
            doc.metadata.get("source", "importado")
            for doc in documents
        ))

        # Salva índice
        vector_store.save(file_names=file_names)

        # Atualiza metadados
        stats = vector_store.get_stats()
        cm.update_context_metadata(
            context_name,
            file_names,
            stats.get("total_documents", 0),
        )

        return {
            "status": "success",
            "message": f"Contexto '{context_name}' importado com sucesso",
            "context_name": context_name,
            "total_documents": len(documents),
            "files": file_names,
            "embedding_provider": req.embedding_provider,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao importar contexto: {str(e)}"
        )


# ==================== VERIFICATION ENDPOINTS ====================

@app.post("/api/verify/extract-reference", response_model=ExtractReferenceResponse, tags=["Verification"])
async def extract_reference(
    file: UploadFile = File(..., description="Base reference document"),
    request_data: str = Form(..., description="JSON request data")
):
    """
    Step 1: Extract reference entities from base document.

    - **file**: Base document (PDF, DOCX, etc.)
    - **extraction_query**: Natural language query (e.g., "list of employee names")
    - **llm_provider**: LLM to use (openai, anthropic)
    - **session_ttl**: Session lifetime in seconds (default: 3600)

    Returns session ID for use in compare-target endpoint.
    """
    try:
        req = ExtractReferenceRequest(**json.loads(request_data))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request_data")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request data: {str(e)}")

    # Validate LLM provider
    if req.llm_provider not in ["openai", "anthropic"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid LLM provider: '{req.llm_provider}'. Use 'openai' or 'anthropic'."
        )

    # Save uploaded file temporarily
    upload_dir = Path("data/temp_uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        # Extract reference entities
        reference = verification_engine.extract_reference(
            base_document_path=str(file_path),
            extraction_query=req.extraction_query,
            llm_provider=req.llm_provider,
            session_ttl=req.session_ttl
        )

        # Clean up temp file
        os.remove(file_path)

        return {
            "session_id": reference.session_id,
            "entity_type": reference.entity_type,
            "entities": reference.entities,
            "total_entities": len(reference.entities),
            "base_document": reference.base_document,
            "expires_at": reference.expires_at.isoformat(),
            "message": f"Successfully extracted {len(reference.entities)} {reference.entity_type}"
        }

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.post("/api/verify/compare-target", response_model=CompareTargetResponse, tags=["Verification"])
async def compare_target(
    files: List[UploadFile] = File(..., description="Target documents to verify"),
    session_id: str = Form(..., description="Session ID from extract-reference"),
    llm_provider: str = Form("openai", description="LLM provider"),
    strictness: float = Form(0.7, description="Match strictness (0.5-1.0)")
):
    """
    Step 2: Compare target documents against reference session.

    - **files**: Target documents to verify
    - **session_id**: Session ID from extract-reference step
    - **llm_provider**: LLM to use (openai, anthropic)
    - **strictness**: Match threshold (0.5=loose, 1.0=strict)

    Returns detailed comparison results per target document.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No target files provided")

    # Validate strictness
    if not (0.5 <= strictness <= 1.0):
        raise HTTPException(status_code=400, detail="Strictness must be between 0.5 and 1.0")

    # Validate LLM provider
    if llm_provider not in ["openai", "anthropic"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid LLM provider: '{llm_provider}'. Use 'openai' or 'anthropic'."
        )

    # Save uploaded files
    upload_dir = Path("data/temp_uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for file in files:
        file_path = upload_dir / file.filename
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append(str(file_path))
        except Exception as e:
            # Cleanup on error
            for fp in saved_files:
                if os.path.exists(fp):
                    os.remove(fp)
            raise HTTPException(status_code=500, detail=f"Failed to save file {file.filename}: {str(e)}")

    try:
        # Perform comparison
        results = verification_engine.compare_targets(
            session_id=session_id,
            target_document_paths=saved_files,
            llm_provider=llm_provider,
            strictness=strictness
        )

        # Calculate summary statistics
        total_verified = sum(1 for r in results if r.status == "verified")
        total_partial = sum(1 for r in results if r.status == "partial_match")
        total_mismatch = sum(1 for r in results if r.status == "mismatch")
        avg_confidence = sum(r.overall_confidence for r in results) / len(results) if results else 0.0

        summary_stats = {
            "total_targets": len(results),
            "verified": total_verified,
            "partial_match": total_partial,
            "mismatch": total_mismatch,
            "average_confidence": round(avg_confidence, 3)
        }

        # Clean up temp files
        for file_path in saved_files:
            if os.path.exists(file_path):
                os.remove(file_path)

        return {
            "session_id": session_id,
            "results": [r.dict() for r in results],
            "summary_statistics": summary_stats,
            "message": f"Compared {len(results)} target document(s)"
        }

    except ValueError as e:
        # Session not found
        for file_path in saved_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        # Other errors
        for file_path in saved_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.get("/api/verify/sessions", tags=["Verification"])
def get_verification_sessions():
    """Get count of active verification sessions."""
    return {
        "active_sessions": verification_engine.session_manager.get_active_sessions(),
        "session_ids": verification_engine.session_manager.list_sessions()
    }


# ==================== ADMIN ENDPOINTS ====================

class AnonymousPromptRequest(BaseModel):
    """Requisição para atualizar prompt anônimo."""
    prompt: str = Field(..., description="Template do prompt. Variáveis: {context_name}, {context_label}, {available_contexts}")


@app.get("/api/admin/rag/anonymous-prompt", tags=["Admin"])
def get_anonymous_prompt_config():
    """
    Obtém o prompt anônimo atual e preview do padrão.

    Retorna:
    - **current_prompt**: Prompt customizado (ou vazio se usa padrão)
    - **default_prompt**: Prompt padrão do sistema
    - **using_default**: Se está usando o padrão
    - **preview**: Preview do prompt com valores de exemplo
    """
    config = _load_rag_config()
    custom_prompt = config.get("anonymous_prompt", "")

    # Gera preview
    preview = get_anonymous_prompt(
        context_name="zangari_website",
    )

    return {
        "current_prompt": custom_prompt,
        "default_prompt": DEFAULT_ANONYMOUS_PROMPT,
        "using_default": not bool(custom_prompt),
        "available_variables": ["{context_name}", "{context_label}", "{available_contexts}"],
        "preview": preview,
    }


@app.put("/api/admin/rag/anonymous-prompt", tags=["Admin"])
def update_anonymous_prompt(req: AnonymousPromptRequest):
    """
    Define prompt customizado para usuários anônimos.

    Variáveis disponíveis no template:
    - **{context_name}**: Nome técnico do contexto (ex: lei_condominios)
    - **{context_label}**: Nome amigável (ex: Lei de Condomínios)
    - **{available_contexts}**: Lista dos contextos públicos disponíveis
    """
    # Valida que o template é formatável
    try:
        req.prompt.format(
            context_name="test",
            context_label="Test",
            available_contexts="- Test (test)",
        )
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Variável inválida no template: {e}. Use apenas: {{context_name}}, {{context_label}}, {{available_contexts}}"
        )

    config = _load_rag_config()
    config["anonymous_prompt"] = req.prompt
    _save_rag_config(config)

    return {
        "status": "success",
        "message": "Prompt anônimo atualizado com sucesso",
        "preview": get_anonymous_prompt(context_name="zangari_website"),
    }


@app.delete("/api/admin/rag/anonymous-prompt", tags=["Admin"])
def reset_anonymous_prompt():
    """
    Reseta o prompt anônimo para o padrão do sistema.
    """
    config = _load_rag_config()
    config.pop("anonymous_prompt", None)
    _save_rag_config(config)

    return {
        "status": "success",
        "message": "Prompt anônimo resetado para o padrão",
        "default_prompt": DEFAULT_ANONYMOUS_PROMPT,
    }


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
