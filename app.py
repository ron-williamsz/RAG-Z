"""RAG Simple - Interface Gradio Multi-Contexto para indexação e consulta de documentos."""

import os
import logging
import httpx
from pathlib import Path
from typing import List, Optional

import gradio as gr
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

from src.document_loader import DocumentLoader
from src.chunker import Chunker
from src.embeddings import EmbeddingsManager
from src.vector_store import VectorStore
from src.rag_chain import RAGChain
from src.context_manager import ContextManager

# Carrega variáveis de ambiente
load_dotenv()

# BDforAll API
BDFORALL_API_URL = os.getenv("BDFORALL_API_URL", "https://api.bdforall.grupozangari.com.br/api")


def authenticate_bdforall(email: str, password: str) -> bool:
    """Autentica usuário via API BDforAll."""
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(
                f"{BDFORALL_API_URL}/auth/login",
                params={"email": email, "senha": password},
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("access_token"):
                    logging.info(f"[AUTH] Login OK: {email}")
                    return True
            logging.warning(f"[AUTH] Login falhou: {email} (status {resp.status_code})")
            return False
    except Exception as e:
        logging.error(f"[AUTH] Erro ao autenticar: {e}")
        return False


# Estado global
class AppState:
    def __init__(self):
        self.vector_store: Optional[VectorStore] = None
        self.rag_chain: Optional[RAGChain] = None
        self.document_loader = DocumentLoader()
        self.chunker = Chunker.from_config("config.toml")
        self.embeddings = EmbeddingsManager.from_config("config.toml")
        self.indexed_files: List[str] = []
        self.current_context: str = "default"
        self.context_manager = ContextManager()
        self.current_embeddings_provider: str = "ollama"  # ollama ou openai


state = AppState()


# ============================================================================
# FUNÇÕES DE GERENCIAMENTO DE CONTEXTOS
# ============================================================================

def get_available_contexts() -> List[str]:
    """Retorna lista de contextos disponíveis."""
    contexts = state.context_manager.list_contexts()
    if not contexts:
        # Cria contexto padrão se não existe nenhum
        state.context_manager.create_context("default", "Contexto padrão")
        contexts = ["default"]
    return contexts


def create_new_context(context_name: str, description: str = "") -> tuple:
    """Cria um novo contexto e o carrega automaticamente."""
    if not context_name or not context_name.strip():
        return "❌ Nome do contexto não pode estar vazio.", gr.update(), gr.update(), gr.update()

    context_name = context_name.strip().lower().replace(" ", "_")

    # Validação de nome
    if not context_name.replace("_", "").replace("-", "").isalnum():
        return "❌ Nome deve conter apenas letras, números, _ ou -", gr.update(), gr.update(), gr.update()

    if state.context_manager.create_context(context_name, description):
        contexts = get_available_contexts()

        # Carrega o contexto automaticamente
        state.current_context = context_name
        state.vector_store = None
        state.rag_chain = None
        state.indexed_files = []

        label = get_current_context_label()

        return (
            f"✅ Contexto '{context_name}' criado e carregado!\n\nAgora você pode indexar documentos.",
            gr.update(choices=contexts, value=context_name),
            label,
            label,
        )
    else:
        return f"❌ Contexto '{context_name}' já existe.", gr.update(), gr.update(), gr.update()


def switch_context(context_name: str) -> str:
    """Muda para um contexto diferente."""
    if not context_name:
        return "❌ Selecione um contexto."

    try:
        state.current_context = context_name

        # Reseta estado
        state.vector_store = None
        state.rag_chain = None
        state.indexed_files = []

        # Verifica se contexto tem índice
        if state.context_manager.has_index(context_name):
            # Carrega índice do contexto
            state.vector_store = VectorStore(
                embeddings=state.embeddings.embeddings,
                context_name=context_name,
            )
            state.vector_store.load()

            state.rag_chain = RAGChain.from_config(
                vector_store=state.vector_store,
                config_path="config.toml",
                llm_provider="openai",
                context_name=context_name,
            )

            state.indexed_files = state.vector_store.indexed_files or []

        # Monta informações
        metadata = state.context_manager.get_context_metadata(context_name)
        info = f"✅ Contexto '{context_name}' carregado!\n\n"

        if metadata:
            total_docs = metadata.get("total_documents", 0)
            files = metadata.get("indexed_files", [])

            if total_docs > 0:
                info += f"📊 {total_docs} documentos indexados\n"
                info += f"📁 {len(files)} arquivos\n"

                if metadata.get("last_updated"):
                    info += f"📅 Última atualização: {metadata['last_updated'][:19]}\n"

                if metadata.get("description"):
                    info += f"📝 Descrição: {metadata['description']}\n"

                info += f"\n📁 Arquivos:\n"
                for name in files[:10]:
                    info += f"  • {name}\n"
                if len(files) > 10:
                    info += f"  ... e mais {len(files) - 10}\n"
            else:
                info += "⚠️ Nenhum documento indexado ainda.\n"
                info += "Adicione documentos na aba 'Indexação'!"
        else:
            info += "⚠️ Nenhum documento indexado ainda.\n"
            info += "Adicione documentos na aba 'Indexação'!"

        return info

    except Exception as e:
        return f"❌ Erro ao carregar contexto: {str(e)}"


def delete_context(context_name: str) -> tuple:
    """Deleta um contexto."""
    if not context_name:
        return "❌ Selecione um contexto.", gr.update(), gr.update(), gr.update()

    if context_name == "default":
        return "❌ Não é possível deletar o contexto padrão.", gr.update(), gr.update(), gr.update()

    if state.context_manager.delete_context(context_name):
        # Reseta se era o contexto atual
        if state.current_context == context_name:
            state.current_context = "default"
            state.vector_store = None
            state.rag_chain = None
            state.indexed_files = []

        contexts = get_available_contexts()
        label = get_current_context_label()
        return (
            f"🗑️ Contexto '{context_name}' deletado. Contexto atual: default",
            gr.update(choices=contexts, value="default"),
            label,
            label,
        )
    else:
        return f"❌ Erro ao deletar contexto '{context_name}'.", gr.update(), gr.update(), gr.update()


def clear_context_index(context_name: str) -> str:
    """Limpa apenas o índice de um contexto (mantém o contexto)."""
    if not context_name:
        return "❌ Selecione um contexto."

    if state.context_manager.clear_context_index(context_name):
        if state.current_context == context_name:
            state.vector_store = None
            state.rag_chain = None
            state.indexed_files = []

        return f"🗑️ Índice do contexto '{context_name}' limpo!\n\nVocê pode adicionar novos documentos."
    else:
        return f"❌ Erro ao limpar índice do contexto."


# ============================================================================
# FUNÇÕES DE INDEXAÇÃO
# ============================================================================

def _process_single_file_in_context(docs: list, file_name: str, context_name: str) -> None:
    """Processa e indexa um único arquivo no contexto especificado."""
    # Verifica se há conteúdo nos documentos
    total_content = sum(len(doc.page_content.strip()) for doc in docs)
    if total_content < 10:
        raise ValueError("Sem texto (OCR também falhou)")

    # Aplica chunking
    chunks = state.chunker.split(docs)

    if not chunks:
        raise ValueError("Conteúdo insuficiente (ignorado)")

    # Inicializa vector store para o contexto
    if state.vector_store is None or state.vector_store.context_name != context_name:
        state.vector_store = VectorStore(
            embeddings=state.embeddings.embeddings,
            context_name=context_name,
        )
    
    # Verifica se mudou o provider de embeddings
    current_provider = state.embeddings.provider
    if state.current_embeddings_provider != current_provider:
        # Recria vector store com novo provider
        state.vector_store = VectorStore(
            embeddings=state.embeddings.embeddings,
            context_name=context_name,
        )
        state.current_embeddings_provider = current_provider

    # Verifica se índice existe
    if not state.vector_store.is_initialized and state.context_manager.has_index(context_name):
        state.vector_store.load()

    # Cria ou adiciona ao índice
    if state.vector_store.is_initialized:
        state.vector_store.add_documents(chunks)
        existing_files = state.vector_store.indexed_files or []
        all_files = list(set(existing_files + [file_name]))
    else:
        state.vector_store.create_index(chunks)
        all_files = [file_name]

    # Salva índice
    state.vector_store.save(file_names=all_files)

    # Atualiza metadados do contexto
    total_stats = state.vector_store.get_stats()
    state.context_manager.update_context_metadata(
        context_name,
        all_files,
        total_stats.get("total_documents", 0),
    )

    # Inicializa RAG Chain se necessário
    if state.rag_chain is None or state.rag_chain.context_name != context_name:
        state.rag_chain = RAGChain.from_config(
            vector_store=state.vector_store,
            config_path="config.toml",
            llm_provider="openai",
            context_name=context_name,
        )

    state.indexed_files = all_files


def index_documents(files, embeddings_choice: str = None) -> str:
    """Indexa documentos no contexto atual."""
    if not files:
        return "❌ Nenhum arquivo selecionado."

    # Usa Ollama como padrão se nada for selecionado (grátis)
    if not embeddings_choice:
        embeddings_choice = "Ollama BGE-M3 (Local - Grátis)"
    
    # Atualiza provider de embeddings
    provider = "ollama" if "Ollama" in embeddings_choice else "openai"
    if provider != state.embeddings.provider:
        state.embeddings = EmbeddingsManager.from_config("config.toml", override_provider=provider)
        state.current_embeddings_provider = provider
        # Reseta vector store para usar novo provider
        state.vector_store = None
        state.rag_chain = None

    context_name = state.current_context
    successful_files = []
    failed_files = []

    for file in files:
        file_path = file.name
        file_name = Path(file_path).name

        try:
            docs = state.document_loader.load(file_path)

            if not docs:
                failed_files.append(f"{file_name} (sem conteúdo)")
                continue

            _process_single_file_in_context(docs, file_name, context_name)
            successful_files.append(file_name)

        except Exception as e:
            failed_files.append(f"{file_name} ({str(e)[:50]})")

    # Monta relatório
    report = [f"📂 **Contexto:** {context_name}"]
    report.append(f"🔧 **Modelo de Embeddings:** {state.embeddings.model} ({state.embeddings.provider.upper()})\\n")

    if successful_files:
        report.append(f"✅ {len(successful_files)} arquivo(s) indexado(s):")
        for name in successful_files[:10]:
            report.append(f"  • {name}")
        if len(successful_files) > 10:
            report.append(f"  ... e mais {len(successful_files) - 10}")

    if failed_files:
        report.append(f"\n❌ {len(failed_files)} arquivo(s) com erro:")
        for name in failed_files[:5]:
            report.append(f"  • {name}")
        if len(failed_files) > 5:
            report.append(f"  ... e mais {len(failed_files) - 5}")

    if state.vector_store and state.vector_store.is_initialized:
        total_stats = state.vector_store.get_stats()
        report.append(f"\n📚 Total no contexto:")
        report.append(f"  • {total_stats.get('total_documents', '?')} chunks")
        report.append(f"  • {total_stats.get('total_files', '?')} arquivos")

    return "\n".join(report)


def index_directory(folder_path: str, recursive: bool = True, embeddings_choice: str = None) -> str:
    """Indexa todos os documentos de uma pasta no contexto atual."""
    if not folder_path or not folder_path.strip():
        return "❌ Por favor, informe o caminho da pasta."
    
    # Usa Ollama como padrão se nada for selecionado (grátis)
    if not embeddings_choice:
        embeddings_choice = "Ollama BGE-M3 (Local - Grátis)"
    
    # Atualiza provider de embeddings
    provider = "ollama" if "Ollama" in embeddings_choice else "openai"
    if provider != state.embeddings.provider:
        state.embeddings = EmbeddingsManager.from_config("config.toml", override_provider=provider)
        state.current_embeddings_provider = provider
        # Reseta vector store para usar novo provider
        state.vector_store = None
        state.rag_chain = None

    folder_path = folder_path.strip()
    context_name = state.current_context
    path = Path(folder_path)

    if not path.exists():
        return f"❌ Pasta não encontrada: {folder_path}"

    if not path.is_dir():
        return f"❌ O caminho não é uma pasta: {folder_path}"

    try:
        all_documents = state.document_loader.load_directory(path, recursive=recursive)

        if not all_documents:
            return f"❌ Nenhum documento suportado encontrado em: {folder_path}"

        # Agrupa documentos por arquivo fonte
        docs_by_file = {}
        for doc in all_documents:
            source = doc.metadata.get("source", "unknown")
            if source not in docs_by_file:
                docs_by_file[source] = []
            docs_by_file[source].append(doc)

        # Processa cada arquivo
        successful_files = []
        failed_files = []

        for file_source, docs in docs_by_file.items():
            file_name = Path(file_source).name
            try:
                _process_single_file_in_context(docs, file_name, context_name)
                successful_files.append(file_name)
            except Exception as e:
                failed_files.append(f"{file_name} ({str(e)[:50]})")

        # Monta relatório
        report = [f"📂 **Contexto:** {context_name}\n"]

        if successful_files:
            report.append(f"✅ {len(successful_files)} arquivo(s) indexado(s):")
            for name in successful_files[:10]:
                report.append(f"  • {name}")
            if len(successful_files) > 10:
                report.append(f"  ... e mais {len(successful_files) - 10}")

        if failed_files:
            report.append(f"\n❌ {len(failed_files)} arquivo(s) com erro:")
            for name in failed_files[:5]:
                report.append(f"  • {name}")

        if state.vector_store and state.vector_store.is_initialized:
            total_stats = state.vector_store.get_stats()
            report.append(f"\n📚 Total no contexto:")
            report.append(f"  • {total_stats.get('total_documents', '?')} chunks")
            report.append(f"  • {total_stats.get('total_files', '?')} arquivos")

        return "\n".join(report)

    except PermissionError:
        return f"❌ Sem permissão para acessar: {folder_path}"
    except Exception as e:
        return f"❌ Erro na indexação: {str(e)}"


# ============================================================================
# FUNÇÕES DE CONSULTA
# ============================================================================

def query_rag(question: str, llm_choice: str) -> tuple:
    """Executa query no contexto atual."""
    if not question.strip():
        return "Por favor, digite uma pergunta.", ""

    if state.rag_chain is None:
        return f"❌ Contexto '{state.current_context}' não tem documentos indexados.", ""

    try:
        provider = "openai" if llm_choice == "GPT-4o (OpenAI)" else "anthropic"
        if state.rag_chain.llm_provider != provider:
            state.rag_chain.switch_llm(provider)

        result = state.rag_chain.query(question, return_sources=True)
        answer = result["answer"]

        sources_text = "📚 **Fontes consultadas:**\n\n"
        for i, source in enumerate(result.get("sources", []), 1):
            sources_text += f"**[{i}] {source['file']}** (chunk {source['chunk']})\n"
            sources_text += f"> {source['content']}\n\n"

        return answer, sources_text

    except Exception as e:
        return f"❌ Erro na consulta: {str(e)}", ""


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def get_current_context_label() -> str:
    """Retorna label do contexto atual."""
    return f"📂 Contexto: **{state.current_context}**"


def get_status() -> str:
    """Retorna status geral do sistema."""
    stats = state.context_manager.get_stats()

    status = f"📊 **Estatísticas Gerais**\n\n"
    status += f"• Total de contextos: {stats['total_contexts']}\n"
    status += f"• Total de documentos: {stats['total_documents']}\n"
    status += f"• Total de arquivos: {stats['total_files']}\n\n"

    if stats['contexts']:
        status += "📂 **Contextos:**\n"
        for ctx in stats['contexts']:
            status += f"  • {ctx['name']}: {ctx['documents']} docs, {ctx['files']} arquivos\n"

    return status


# ============================================================================
# INTERFACE GRADIO
# ============================================================================

with gr.Blocks(
    title="RAG Multi-Contexto",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown("""
    # 🔍 RAG Multi-Contexto
    **Gerencie documentos por contexto (ex: cond_169, cond_170)**

    Cada contexto tem seu próprio índice independente!
    """)

    with gr.Row():
        # =====================================================================
        # COLUNA ESQUERDA - Gerenciamento de Contextos
        # =====================================================================
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Contextos")

            context_dropdown = gr.Dropdown(
                choices=get_available_contexts(),
                value=state.current_context,
                label="Selecione o Contexto",
                interactive=True,
            )

            with gr.Row():
                load_context_btn = gr.Button("📥 Carregar", variant="primary", scale=2)
                refresh_btn = gr.Button("🔄", scale=1)

            context_status = gr.Textbox(
                label="Status do Contexto",
                lines=12,
                interactive=False,
            )

            gr.Markdown("---")
            gr.Markdown("**Criar Novo Contexto:**")

            new_context_name = gr.Textbox(
                label="Nome",
                placeholder="ex: cond_169",
            )
            new_context_desc = gr.Textbox(
                label="Descrição (opcional)",
                placeholder="Documentos do condomínio 169",
            )
            create_btn = gr.Button("➕ Criar Contexto", variant="secondary")

            gr.Markdown("---")
            gr.Markdown("**Gerenciar:**")

            with gr.Row():
                clear_index_btn = gr.Button("🗑️ Limpar Índice", variant="stop", scale=1)
                delete_btn = gr.Button("❌ Deletar", variant="stop", scale=1)

        # =====================================================================
        # COLUNA CENTRAL - Indexação
        # =====================================================================
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Indexação")

            current_context_label = gr.Markdown(get_current_context_label())
            
            with gr.Tab("Upload"):
                file_input = gr.File(
                    label="Selecione documentos",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".doc", ".xlsx", ".xls", ".txt", ".md"],
                )
                index_btn = gr.Button("🔄 Indexar", variant="primary")

            with gr.Tab("Pasta / Rede"):
                folder_input = gr.Textbox(
                    label="Caminho da Pasta",
                    placeholder="Ex: G:\\... ou \\\\servidor\\pasta",
                )
                recursive_check = gr.Checkbox(label="Incluir subpastas", value=True)
                index_folder_btn = gr.Button("📂 Indexar Pasta", variant="primary")

            index_output = gr.Textbox(
                label="Resultado da Indexação",
                lines=10,
                interactive=False,
            )

        # =====================================================================
        # COLUNA DIREITA - Consulta
        # =====================================================================
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Consulta")

            query_context_label = gr.Markdown(get_current_context_label())

            with gr.Row():
                llm_choice = gr.Dropdown(
                    choices=["GPT-4o (OpenAI)", "Claude Sonnet (Anthropic)"],
                    value="GPT-4o (OpenAI)",
                    label="Modelo LLM (Consulta)",
                    scale=1,
                    interactive=True,
                    allow_custom_value=False,
                )
                embeddings_choice = gr.Dropdown(
                    choices=["Ollama BGE-M3 (Local - Grátis)", "OpenAI text-embedding-3-small (Pago)"],
                    value=None,
                    label="Modelo de Embeddings (Indexação) - Padrão: Ollama Grátis",
                    scale=1,
                    interactive=True,
                    allow_custom_value=False,
                )

            question_input = gr.Textbox(
                label="Sua pergunta",
                placeholder="Digite sua pergunta sobre os documentos...",
                lines=2,
            )

            query_btn = gr.Button("🔍 Buscar Resposta", variant="primary")

            answer_output = gr.Textbox(
                label="Resposta",
                lines=10,
                interactive=False,
            )

            sources_output = gr.Markdown(label="Fontes")

    # Status geral
    with gr.Row():
        status_btn = gr.Button("📊 Status Geral")
        status_output = gr.Textbox(label="Status", lines=6, interactive=False)

    # Timer para auto-refresh do dropdown (sincroniza contextos criados via API)
    dropdown_timer = gr.Timer(value=30)

    # =========================================================================
    # EVENTOS
    # =========================================================================

    def refresh_dropdown():
        contexts = get_available_contexts()
        return gr.update(choices=contexts, value=state.current_context)

    def update_context_labels():
        label = get_current_context_label()
        return label, label

    def on_context_change(context_name):
        """Carrega contexto automaticamente ao selecionar no dropdown."""
        if context_name and context_name != state.current_context:
            status = switch_context(context_name)
            label = get_current_context_label()
            return status, label, label, ""  # Limpa resultado anterior
        return gr.update(), gr.update(), gr.update(), gr.update()

    # Contextos - Carrega automaticamente ao selecionar
    context_dropdown.change(
        fn=on_context_change,
        inputs=[context_dropdown],
        outputs=[context_status, current_context_label, query_context_label, index_output],
    )

    refresh_btn.click(
        fn=refresh_dropdown,
        outputs=[context_dropdown],
    )

    load_context_btn.click(
        fn=switch_context,
        inputs=[context_dropdown],
        outputs=[context_status],
    ).then(
        fn=update_context_labels,
        outputs=[current_context_label, query_context_label],
    )

    create_btn.click(
        fn=create_new_context,
        inputs=[new_context_name, new_context_desc],
        outputs=[context_status, context_dropdown, current_context_label, query_context_label],
    )

    delete_btn.click(
        fn=delete_context,
        inputs=[context_dropdown],
        outputs=[context_status, context_dropdown, current_context_label, query_context_label],
    )

    clear_index_btn.click(
        fn=clear_context_index,
        inputs=[context_dropdown],
        outputs=[context_status],
    )

    # Indexação
    index_btn.click(
        fn=index_documents,
        inputs=[file_input, embeddings_choice],
        outputs=[index_output],
    )

    index_folder_btn.click(
        fn=index_directory,
        inputs=[folder_input, recursive_check, embeddings_choice],
        outputs=[index_output],
    )

    # Consulta
    query_btn.click(
        fn=query_rag,
        inputs=[question_input, llm_choice],
        outputs=[answer_output, sources_output],
    )

    question_input.submit(
        fn=query_rag,
        inputs=[question_input, llm_choice],
        outputs=[answer_output, sources_output],
    )

    # Status + refresh do dropdown
    status_btn.click(
        fn=get_status,
        outputs=[status_output],
    ).then(
        fn=refresh_dropdown,
        outputs=[context_dropdown],
    )

    # Auto-refresh do dropdown a cada 30s (sincroniza contextos criados via API)
    dropdown_timer.tick(
        fn=refresh_dropdown,
        outputs=[context_dropdown],
    )

    # =========================================================================
    # VERIFICATION SECTION
    # =========================================================================

    gr.Markdown("---")
    gr.Markdown("""
    ## 🔍 Verificação de Documentos

    Compare documentos para verificar consistência de entidades (ex: listas de funcionários, números de nota fiscal).

    **Fluxo:** Extrair entidades de referência → Enviar documentos alvo → Visualizar resultados da comparação
    """)

    with gr.Row():
        # LEFT COLUMN: Reference Extraction
        with gr.Column(scale=1):
            gr.Markdown("#### Passo 1: Extrair Referência")

            verify_base_doc = gr.File(
                label="Documento Base de Referência",
                file_types=[".pdf", ".docx", ".xlsx", ".txt"],
                file_count="single"
            )

            verify_extraction_query = gr.Textbox(
                label="O que extrair?",
                placeholder="ex: 'lista de nomes de funcionários', 'números de notas fiscais'",
                lines=2
            )

            verify_llm_extract = gr.Dropdown(
                choices=["GPT-4o (OpenAI)", "Claude Sonnet (Anthropic)"],
                value="GPT-4o (OpenAI)",
                label="LLM para Extração"
            )

            verify_extract_btn = gr.Button("🔍 Extrair Referência", variant="primary")

            verify_session_output = gr.Textbox(
                label="Informações da Sessão",
                lines=8,
                interactive=False
            )

            verify_session_id = gr.State("")  # Hidden state for session ID

        # RIGHT COLUMN: Target Comparison
        with gr.Column(scale=1):
            gr.Markdown("#### Passo 2: Comparar Alvos")

            verify_target_docs = gr.File(
                label="Documentos Alvo",
                file_types=[".pdf", ".docx", ".xlsx", ".txt"],
                file_count="multiple"
            )

            verify_target_query = gr.Textbox(
                label="Query para Documento Alvo (opcional)",
                placeholder="Deixe vazio para usar a mesma query do Passo 1, ou especifique outra query",
                lines=2,
                info="Se os documentos têm formatos diferentes, você pode especificar uma query diferente aqui"
            )

            verify_strictness = gr.Slider(
                minimum=0.5,
                maximum=1.0,
                value=0.7,
                step=0.05,
                label="Rigor da Correspondência",
                info="0.5 = flexível (aceita variações), 1.0 = rigoroso (apenas exato)"
            )

            verify_llm_compare = gr.Dropdown(
                choices=["GPT-4o (OpenAI)", "Claude Sonnet (Anthropic)"],
                value="GPT-4o (OpenAI)",
                label="LLM para Comparação"
            )

            verify_compare_btn = gr.Button("⚖️ Comparar", variant="secondary")

            verify_comparison_output = gr.Textbox(
                label="Resultados da Comparação",
                lines=8,
                interactive=False
            )

    # Full-width detailed results
    with gr.Row():
        verify_detailed_results = gr.Markdown("")

    # Verification event handlers
    def verify_extract_reference_entities(base_doc, query, llm_choice):
        """Handler for reference extraction."""
        from pathlib import Path
        from datetime import datetime, timedelta

        if not base_doc:
            return "❌ Por favor, envie um documento base de referência.", ""

        if not query or not query.strip():
            return "❌ Por favor, especifique o que extrair.", ""

        try:
            # Map LLM choice to provider
            provider = "openai" if "OpenAI" in llm_choice else "anthropic"

            # Call verification engine
            from src.verification_engine import VerificationEngine
            engine = VerificationEngine()

            reference = engine.extract_reference(
                base_document_path=base_doc.name,
                extraction_query=query,
                llm_provider=provider,
                session_ttl=3600
            )

            result_text = f"""✅ Extração de Referência Completa!

📄 **Documento Base:** {reference.base_document}
🔍 **Query:** {query}
🤖 **LLM:** {llm_choice}

📊 **Entidades Extraídas ({len(reference.entities)}):**
"""
            # Show first 20 entities
            for i, entity in enumerate(reference.entities[:20], 1):
                result_text += f"\n{i}. {entity}"

            if len(reference.entities) > 20:
                result_text += f"\n... e mais {len(reference.entities) - 20}"

            expires_at = datetime.fromisoformat(reference.expires_at.replace('Z', '+00:00')) if isinstance(reference.expires_at, str) else reference.expires_at
            result_text += f"""

⏱️ **Sessão expira em:** 1 hora
🔑 **ID da Sessão:** {reference.session_id[:16]}...

✅ Você já pode enviar os documentos alvo para comparar."""

            return result_text, reference.session_id

        except Exception as e:
            return f"❌ Extração falhou: {str(e)}", ""

    def verify_compare_target_documents(target_docs, target_query, session_id, strictness, llm_choice):
        """Handler for target comparison."""
        from pathlib import Path

        if not target_docs:
            return "❌ Por favor, envie documentos alvo.", ""

        if not session_id:
            return "❌ Por favor, extraia as entidades de referência primeiro (Passo 1).", ""

        try:
            # Map LLM choice to provider
            provider = "openai" if "OpenAI" in llm_choice else "anthropic"

            # Call verification engine
            from src.verification_engine import VerificationEngine
            engine = VerificationEngine()

            # Get target file paths
            target_paths = [doc.name for doc in target_docs] if isinstance(target_docs, list) else [target_docs.name]

            # Use target_query if provided, otherwise None (will use reference query)
            target_extraction_query = target_query.strip() if target_query and target_query.strip() else None

            results = engine.compare_targets(
                session_id=session_id,
                target_document_paths=target_paths,
                llm_provider=provider,
                strictness=strictness,
                target_extraction_query=target_extraction_query
            )

            # Calculate summary
            total_verified = sum(1 for r in results if r.status == "verified")
            total_partial = sum(1 for r in results if r.status == "partial_match")
            total_mismatch = sum(1 for r in results if r.status == "mismatch")
            avg_confidence = sum(r.overall_confidence for r in results) / len(results) if results else 0.0

            results_text = f"""✅ Comparação Completa!

📊 **Resumo:**
• Total de alvos: {len(results)}
• ✅ Verificados: {total_verified}
• ⚠️ Correspondência parcial: {total_partial}
• ❌ Não correspondente: {total_mismatch}
• Confiança média: {avg_confidence:.1%}

Rigor: {strictness:.2f}

📋 Veja os resultados detalhados abaixo."""

            # Build detailed markdown
            detailed = "## Resultados Detalhados\n\n"

            for i, result in enumerate(results, 1):
                status_emoji = {"verified": "✅", "partial_match": "⚠️", "mismatch": "❌"}[result.status]

                detailed += f"### {i}. {result.target_document}\n"
                detailed += f"**Status:** {status_emoji} {result.summary}\n\n"

                # Show total entities extracted from target
                total_target_entities = len(result.extracted_target_entities)
                detailed += f"**📊 Entidades extraídas do alvo:** {total_target_entities}\n"
                detailed += f"**📊 Entidades na referência:** {len(result.matched_entities)}\n\n"

                # Show ALL extracted entities from target (for debugging)
                if result.extracted_target_entities:
                    detailed += f"<details><summary>🔍 <b>Ver todas as {len(result.extracted_target_entities)} entidades extraídas do alvo</b></summary>\n\n"
                    for i, entity in enumerate(result.extracted_target_entities, 1):
                        detailed += f"{i}. {entity}\n"
                    detailed += "\n</details>\n\n"

                # Show matched entities
                matched = [m for m in result.matched_entities if m.match_type != "no_match"]
                if matched:
                    detailed += "**Entidades Correspondentes:**\n"
                    for m in matched[:10]:  # Show first 10
                        match_emoji = "✅" if m.match_type == "exact" else "🔄"
                        detailed += f"- {match_emoji} {m.reference_entity}"
                        if m.target_entity != m.reference_entity:
                            detailed += f" → {m.target_entity}"
                        detailed += f" ({m.confidence:.0%})\n"
                    if len(matched) > 10:
                        detailed += f"... e mais {len(matched) - 10}\n"
                    detailed += "\n"

                # Show missing entities
                if result.missing_in_target:
                    detailed += "**Faltando no alvo:**\n"
                    for entity in result.missing_in_target[:10]:
                        detailed += f"- ❌ {entity}\n"
                    if len(result.missing_in_target) > 10:
                        detailed += f"... e mais {len(result.missing_in_target) - 10}\n"
                    detailed += "\n"

                # Show extra entities
                if result.extra_in_target:
                    detailed += "**Extras no alvo (não estão na referência):**\n"
                    for entity in result.extra_in_target[:10]:
                        detailed += f"- ➕ {entity}\n"
                    if len(result.extra_in_target) > 10:
                        detailed += f"... e mais {len(result.extra_in_target) - 10}\n"
                    detailed += "\n"

                detailed += "---\n\n"

            return results_text, detailed

        except ValueError as e:
            # Session expired or not found
            return f"❌ {str(e)}\n\nPor favor, extraia as entidades de referência novamente (Passo 1).", ""
        except Exception as e:
            return f"❌ Comparação falhou: {str(e)}", ""

    # Wire up verification events
    verify_extract_btn.click(
        fn=verify_extract_reference_entities,
        inputs=[verify_base_doc, verify_extraction_query, verify_llm_extract],
        outputs=[verify_session_output, verify_session_id]
    )

    verify_compare_btn.click(
        fn=verify_compare_target_documents,
        inputs=[verify_target_docs, verify_target_query, verify_session_id, verify_strictness, verify_llm_compare],
        outputs=[verify_comparison_output, verify_detailed_results]
    )

    gr.Markdown("""
    ---
    **Formatos suportados:** PDF (com OCR), DOCX, XLSX, TXT, MD

    **Tecnologias:** LangChain • FAISS • OCR • OpenAI • Anthropic
    """)


if __name__ == "__main__":
    # Cria diretórios necessários
    Path("data/documents").mkdir(parents=True, exist_ok=True)
    Path("data/faiss_index").mkdir(parents=True, exist_ok=True)

    # Garante contexto padrão existe
    if not state.context_manager.context_exists("default"):
        state.context_manager.create_context("default", "Contexto padrão")

    # Tenta carregar contexto padrão se tiver índice
    if state.context_manager.has_index("default"):
        switch_context("default")

    # Inicia aplicação
    demo.queue(
        default_concurrency_limit=1,  # 1 indexação por vez (CPU)
        max_size=10,                  # fila de até 10 requisições
    )
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        auth=authenticate_bdforall,
        auth_message="🔒 RAG Multi-Contexto — Login com suas credenciais BDforAll",
    )
