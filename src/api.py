"""API REST para o sistema RAG usando FastAPI."""

import os
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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


# ==================== MODELS ====================

class QueryRequest(BaseModel):
    """Modelo de requisição para consultas."""
    question: str = Field(..., description="Pergunta do usuário", min_length=1)
    context: str = Field("default", description="Nome do contexto a usar")
    llm_provider: str = Field("openai", description="Provedor do LLM (openai, anthropic)")
    embedding_provider: str = Field("ollama", description="Provedor de embeddings (ollama, openai)")
    top_k: int = Field(8, description="Número de chunks a recuperar", ge=1, le=20)
    return_sources: bool = Field(True, description="Se deve retornar fontes")
    use_legal_hierarchy: bool = Field(False, description="Usar hierarquia legal (codigo_civil > lei_condominios > contexto)")


class HierarchicalQueryRequest(BaseModel):
    """Modelo de requisição para consultas com hierarquia legal."""
    question: str = Field(..., description="Pergunta do usuário", min_length=1)
    context: str = Field(..., description="Contexto do condomínio (ex: cond_0388)")
    llm_provider: str = Field("openai", description="Provedor do LLM (openai, anthropic)")
    embedding_provider: str = Field("ollama", description="Provedor de embeddings (ollama, openai)")
    top_k_per_context: int = Field(3, description="Chunks por contexto hierárquico", ge=1, le=10)
    return_sources: bool = Field(True, description="Se deve retornar fontes")


class HierarchicalQueryResponse(BaseModel):
    """Modelo de resposta para consultas hierárquicas."""
    answer: str
    sources: Optional[List[dict]] = None
    contexts_searched: List[str]
    hierarchy_applied: bool
    llm_provider: str
    embedding_provider: str
    context_format: Optional[str] = None


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


# ==================== APP ====================

app = FastAPI(
    title="RAG API",
    description="API REST para sistema RAG (Retrieval-Augmented Generation)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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

# ==================== HELPERS ====================

# Hierarquia legal para busca em múltiplos contextos
# Ordem: 1º Código Civil, 2º Lei de Condomínios, 3º Contexto do Condomínio
LEGAL_HIERARCHY_CONTEXTS = ["codigo_civil", "lei_condominios"]


def _get_context_manager() -> ContextManager:
    """Retorna instância do ContextManager."""
    return ContextManager()


def _search_with_hierarchy(
    question: str,
    condo_context: str,
    embedding_provider: str,
    top_k_per_context: int = 3,
) -> tuple[list, list[str]]:
    """
    Busca em múltiplos contextos respeitando hierarquia legal.

    Hierarquia:
    1º codigo_civil (Lei 10.406/2002) - SUPERIOR
    2º lei_condominios (Lei 4.591/64)
    3º contexto do condomínio (convenção, regimento, atas)

    Args:
        question: Pergunta do usuário
        condo_context: Contexto do condomínio (ex: cond_0388)
        embedding_provider: Provedor de embeddings
        top_k_per_context: Número de chunks por contexto

    Returns:
        Tuple de (documentos combinados, contextos encontrados)
    """
    cm = _get_context_manager()
    available_contexts = cm.list_contexts()

    # Lista de contextos na ordem hierárquica
    contexts_to_search = LEGAL_HIERARCHY_CONTEXTS + [condo_context]

    all_documents = []
    contexts_found = []

    # Cria embeddings manager uma vez
    embeddings_manager = EmbeddingsManager.from_config(
        config_path="config.toml",
        override_provider=embedding_provider,
    )

    for ctx in contexts_to_search:
        if ctx not in available_contexts:
            continue

        try:
            # Cria vector store para este contexto
            vector_store = VectorStore(
                embeddings=embeddings_manager.embeddings,
                context_name=ctx,
            )
            vector_store.load()

            # Busca documentos relevantes
            docs = vector_store.search_documents(
                query=question,
                top_k=top_k_per_context,
            )

            # Adiciona metadado de hierarquia
            hierarchy_level = contexts_to_search.index(ctx) + 1
            hierarchy_names = {
                "codigo_civil": "Código Civil (Lei 10.406/2002)",
                "lei_condominios": "Lei de Condomínios (Lei 4.591/64)",
            }

            for doc in docs:
                doc.metadata["hierarchy_level"] = hierarchy_level
                doc.metadata["hierarchy_context"] = ctx
                doc.metadata["hierarchy_name"] = hierarchy_names.get(ctx, f"Documentos do {ctx}")

            all_documents.extend(docs)
            contexts_found.append(ctx)

        except FileNotFoundError:
            # Contexto existe mas não tem índice
            continue
        except Exception:
            continue

    return all_documents, contexts_found


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
    Executa consulta com hierarquia legal (recomendado para questões condominiais).

    Busca automaticamente em múltiplos contextos respeitando a hierarquia:
    1️⃣ **Código Civil** (Lei 10.406/2002) - Norma suprema
    2️⃣ **Lei de Condomínios** (Lei 4.591/64) - Segunda na hierarquia
    3️⃣ **Documentos do Condomínio** (Convenção, Regimento, Atas)

    - **question**: Pergunta do usuário
    - **context**: Contexto do condomínio (ex: cond_0388)
    - **top_k_per_context**: Chunks por contexto (padrão: 3)
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
        # Busca em múltiplos contextos com hierarquia
        documents, contexts_found = _search_with_hierarchy(
            question=req.question,
            condo_context=req.context,
            embedding_provider=req.embedding_provider,
            top_k_per_context=req.top_k_per_context,
        )

        if not documents:
            raise HTTPException(
                status_code=404,
                detail=f"Nenhum documento encontrado nos contextos: {LEGAL_HIERARCHY_CONTEXTS + [req.context]}. Verifique se os contextos existem e estão indexados."
            )

        # Cria embeddings manager para RAG chain
        embeddings_manager = EmbeddingsManager.from_config(
            config_path="config.toml",
            override_provider=req.embedding_provider,
        )

        # Cria vector store temporário com documentos combinados
        from .toon_formatter import ToonFormatter
        toon_formatter = ToonFormatter(use_toon=True)

        # Formata contexto com indicação de hierarquia
        context_parts = []
        for doc in documents:
            hierarchy_name = doc.metadata.get("hierarchy_name", "Documento")
            level = doc.metadata.get("hierarchy_level", 0)
            source = doc.metadata.get("source", "unknown")

            # Adiciona marcador de hierarquia
            hierarchy_marker = "⚖️" if level <= 2 else "📄"
            context_parts.append(
                f"{hierarchy_marker} [{hierarchy_name}] (Fonte: {source})\n{doc.page_content}"
            )

        combined_context = "\n\n" + "─" * 50 + "\n\n".join(context_parts)

        # Cria LLM
        import os
        if req.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.3,
                max_tokens=4096,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        else:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model="claude-sonnet-4-20250514",
                temperature=0.3,
                max_tokens=4096,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )

        # Usa o prompt do RAGChain com hierarquia
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = ChatPromptTemplate.from_template(RAGChain.DEFAULT_PROMPT_TEMPLATE)
        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({
            "context": combined_context,
            "question": req.question,
        })

        # Prepara fontes com informação de hierarquia
        sources = None
        if req.return_sources:
            sources = [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "file": doc.metadata.get("source", "unknown"),
                    "hierarchy_context": doc.metadata.get("hierarchy_context", "unknown"),
                    "hierarchy_name": doc.metadata.get("hierarchy_name", "Documento"),
                    "hierarchy_level": doc.metadata.get("hierarchy_level", 0),
                }
                for doc in documents
            ]

        return {
            "answer": response,
            "sources": sources,
            "contexts_searched": contexts_found,
            "hierarchy_applied": True,
            "llm_provider": req.llm_provider,
            "embedding_provider": req.embedding_provider,
            "context_format": "toon",
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
):
    """
    Indexa documentos em um contexto específico.

    - **files**: Arquivos para indexar (PDF, DOCX, XLSX, TXT, MD)
    - **context**: Nome do contexto (padrão: "default")
    - **embedding_provider**: Provedor de embeddings (ollama, openai)
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

        for file_path in saved_files:
            docs = loader.load(file_path)
            all_documents.extend(docs)
            file_names.append(Path(file_path).name)

        # Chunking
        chunker = Chunker.from_config("config.toml")
        chunks = chunker.split_documents(all_documents)

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
        cm.update_context_metadata(context, file_names)

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
                stats = cm.get_context_stats(context_name)
                context_infos.append({
                    "name": context_name,
                    "total_files": stats.get("total_files", 0),
                    "total_chunks": stats.get("total_chunks", 0),
                    "created_at": stats.get("created_at"),
                    "last_updated": stats.get("last_updated"),
                })
            except Exception:
                # Se houver erro ao pegar stats, adiciona info mínima
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

        # Obtém estatísticas
        stats = cm.get_context_stats(context_name)

        return stats

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao obter estatísticas: {str(e)}"
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


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
