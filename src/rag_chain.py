"""RAG Chain - Pipeline principal de Retrieval-Augmented Generation."""

import os
import logging
from typing import List, Optional, Literal, Dict, Any

import toml
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from .vector_store import VectorStore
from .toon_formatter import ToonFormatter

logger = logging.getLogger(__name__)

LLMProvider = Literal["openai", "anthropic"]


class RAGChain:
    """Pipeline completo de RAG com suporte a múltiplos LLMs."""

    DEFAULT_PROMPT_TEMPLATE = """Você é um assistente jurídico especializado em direito condominial brasileiro.

<<BRAND_PERSONALITY>>

Sua função é ajudar os usuários a entender as informações contidas nos documentos disponíveis de forma clara, completa e útil.

═══════════════════════════════════════════════════════════════
⚖️ HIERARQUIA DAS NORMAS CONDOMINIAIS (ORDEM DE PREVALÊNCIA):
═══════════════════════════════════════════════════════════════

IMPORTANTE: Sempre respeite esta hierarquia ao responder. Uma norma superior PREVALECE sobre as inferiores em caso de conflito:

1️⃣ **CÓDIGO CIVIL** (Lei 10.406/2002) - Norma suprema
2️⃣ **LEI DE CONDOMÍNIOS** (Lei 4.591/64) - Segunda na hierarquia
3️⃣ **CONVENÇÃO DO CONDOMÍNIO** - Terceira na hierarquia
4️⃣ **REGIMENTO INTERNO** - Quarta na hierarquia
5️⃣ **DECISÕES DE ASSEMBLEIA** - Última na hierarquia

Se houver conflito entre normas, a de hierarquia SUPERIOR sempre prevalece.
Exemplo: Se o Regimento Interno proíbe algo que o Código Civil permite, prevalece o Código Civil.

═══════════════════════════════════════════════════════════════
DOCUMENTOS DISPONÍVEIS:
═══════════════════════════════════════════════════════════════
{context}
═══════════════════════════════════════════════════════════════

PERGUNTA DO USUÁRIO: {question}

═══════════════════════════════════════════════════════════════
INSTRUÇÕES PARA RESPONDER:
═══════════════════════════════════════════════════════════════

1. **ANALISE COM ATENÇÃO** - Leia todo o contexto. Mesmo que a pergunta não tenha resposta direta e literal, procure informações relacionadas que possam ajudar.

2. **RESPEITE A HIERARQUIA** - Ao citar normas, indique sua posição na hierarquia. Se houver conflito entre documentos, explique qual prevalece e por quê.

3. **CITE AS FONTES COM HIERARQUIA** - Sempre mencione de qual documento veio cada informação:
   - "De acordo com o Código Civil (norma superior)..."
   - "A Lei de Condomínios estabelece que..."
   - "A Convenção do Condomínio prevê..."
   - "O Regimento Interno determina..."
   - "A Assembleia decidiu que..."

4. **ALERTE SOBRE CONFLITOS** - Se identificar conflito entre normas de diferentes hierarquias, SEMPRE informe ao usuário qual prevalece.

5. **SEJA COMPLETO E ÚTIL** - Não dê respostas curtas ou superficiais. Elabore uma resposta rica explicando o que os documentos dizem sobre o assunto.

6. **CONTEXTUALIZE** - Ajude o usuário a entender o contexto jurídico e as implicações práticas. Explique o "porquê" quando possível.

7. **ORGANIZE BEM** - Se houver múltiplas informações:
   - Use parágrafos bem estruturados
   - Organize em tópicos quando apropriado
   - Destaque pontos importantes

8. **QUANDO NÃO HOUVER INFORMAÇÃO** - Apenas se realmente NÃO existir NENHUMA informação relevante no contexto, informe educadamente. Mas sempre tente ajudar com o que está disponível antes de desistir.

IMPORTANTE: Responda em português brasileiro. Seja didático e acessível.

RESPOSTA:"""

    # Template com histórico de conversa (modo técnico)
    PROMPT_WITH_HISTORY_TEMPLATE = """Você é um assistente jurídico especializado em direito condominial brasileiro.

<<BRAND_PERSONALITY>>

{conversation_history}

═══════════════════════════════════════════════════════════════
⚖️ HIERARQUIA DAS NORMAS CONDOMINIAIS (ORDEM DE PREVALÊNCIA):
═══════════════════════════════════════════════════════════════

Se houver conflito entre normas, a de hierarquia SUPERIOR sempre prevalece:
1️⃣ CÓDIGO CIVIL (Lei 10.406/2002) - Norma suprema
2️⃣ LEI DE CONDOMÍNIOS (Lei 4.591/64)
3️⃣ CONVENÇÃO DO CONDOMÍNIO
4️⃣ REGIMENTO INTERNO
5️⃣ DECISÕES DE ASSEMBLEIA

═══════════════════════════════════════════════════════════════
DOCUMENTOS ENCONTRADOS:
═══════════════════════════════════════════════════════════════
{context}
═══════════════════════════════════════════════════════════════

PERGUNTA: {question}

═══════════════════════════════════════════════════════════════
INSTRUÇÕES:
═══════════════════════════════════════════════════════════════

1. **USE APENAS OS DOCUMENTOS FORNECIDOS** - Responda com base APENAS nos documentos acima. NÃO mencione documentos que não foram fornecidos.

2. **CITE AS FONTES DISPONÍVEIS** - Mencione de qual documento veio cada informação:
   - "De acordo com o Código Civil..."
   - "A Lei de Condomínios estabelece que..."
   - "A Convenção prevê..."
   - "O Regimento Interno determina..."

3. **SE HOUVER CONFLITO** - Se identificar conflito entre normas nos documentos fornecidos, explique qual prevalece.

4. **SEJA OBJETIVO** - Vá direto ao ponto, apresente as informações relevantes e conclua.

5. **CONCLUSÃO** - Termine com uma conclusão clara sobre a resposta.

6. **QUANDO NÃO HOUVER INFORMAÇÃO** - Se os documentos fornecidos não contêm informação suficiente: "{not_found_message}"

IMPORTANTE: NÃO mencione que "não encontrou no Código Civil" ou em qualquer outro documento que não foi fornecido. Foque apenas no que está disponível.

RESPOSTA:"""

    # ═══════════════════════════════════════════════════════════════
    # PROMPT FLUIDO - Resposta clara sem mencionar hierarquia técnica
    # ═══════════════════════════════════════════════════════════════

    FLUENT_PROMPT_TEMPLATE = """Você é um assistente prestativo do Grupo Zangari.

<<BRAND_PERSONALITY>>

{conversation_history}

DOCUMENTOS ENCONTRADOS:
{context}

PERGUNTA: {question}

INSTRUÇÕES IMPORTANTES:
1. Responda de forma clara, direta e amigável
2. Use APENAS as informações dos documentos fornecidos acima
3. NÃO mencione termos técnicos como "hierarquia", "precedência", "norma superior"
4. NÃO mencione documentos que NÃO foram fornecidos - foque apenas no que está disponível
5. Se os documentos não contêm informação suficiente para responder, diga: "{not_found_message}"
6. Seja conciso mas completo
7. Termine com uma conclusão clara e objetiva

RESPOSTA:"""

    FLUENT_WITH_SOURCE_PROMPT_TEMPLATE = """Você é um assistente prestativo do Grupo Zangari.

<<BRAND_PERSONALITY>>

{conversation_history}

DOCUMENTOS ENCONTRADOS:
{context}

PERGUNTA: {question}

INSTRUÇÕES IMPORTANTES:
1. Responda de forma clara e amigável
2. Use APENAS as informações dos documentos fornecidos acima
3. MENCIONE A FONTE de forma natural ao citar informações:
   - "De acordo com a convenção do seu condomínio..."
   - "Conforme o regimento interno..."
   - "A Lei de Condomínios estabelece que..."
   - "O Código Civil prevê que..."
4. NÃO mencione documentos que NÃO foram fornecidos - cite apenas os que estão disponíveis
5. Se os documentos não contêm informação suficiente, diga: "{not_found_message}"
6. Termine com uma conclusão clara e objetiva

RESPOSTA:"""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: LLMProvider = "openai",
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        top_k: int = 8,
        use_toon: bool = True,
        system_context: str = "documentos e informações disponíveis",
        context_name: Optional[str] = None,
        brand_personality: str = "",
    ):
        """
        Inicializa o RAG Chain.

        Args:
            vector_store: VectorStore inicializado
            llm_provider: Provedor do LLM ("openai" ou "anthropic")
            model: Nome do modelo (usa padrão se não especificado)
            temperature: Temperatura para geração
            max_tokens: Máximo de tokens na resposta
            top_k: Número de documentos a recuperar
            use_toon: Se True, usa TOON para formatar contexto
            system_context: Descrição do tipo de documentos (personalizável)
            context_name: Nome do contexto atual (ex: cond_169)
        """
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.top_k = top_k
        self.context_name = context_name or "default"
        self.toon_formatter = ToonFormatter(use_toon=use_toon)

        # Se tem context_name, personaliza o system_context
        if context_name and context_name != "default":
            system_context = f"documentos do contexto '{context_name}': {system_context}"

        self.system_context = system_context
        # Substitui {context_name} no brand_personality.
        # Para contextos cond_XXXX, usa o nome; caso contrario usa "o condominio atendido".
        ctx = self.context_name
        if ctx.startswith("cond_"):
            friendly = ctx  # cond_0006 (o nome real e resolvido no widget)
        else:
            friendly = "o condominio atendido"
        self.brand_personality = (brand_personality or "").replace(
            "{context_name}", friendly
        )

        # Configura LLM
        self._llm = self._create_llm(
            provider=llm_provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Configura prompt com o contexto do sistema e personalidade da marca
        prompt_text = self.DEFAULT_PROMPT_TEMPLATE.replace(
            "{system_context}", system_context
        ).replace("<<BRAND_PERSONALITY>>", self.brand_personality)
        self._prompt = ChatPromptTemplate.from_template(prompt_text)
        self._output_parser = StrOutputParser()

    def _inject_personality(self, template: str) -> str:
        """Substitui o placeholder de personalidade no template."""
        return template.replace("<<BRAND_PERSONALITY>>", self.brand_personality)

    @classmethod
    def from_config(
        cls,
        vector_store: VectorStore,
        config_path: str = "config.toml",
        llm_provider: LLMProvider = "openai",
        context_name: Optional[str] = None,
    ) -> "RAGChain":
        """
        Cria RAGChain a partir de arquivo de configuração TOML.

        Args:
            vector_store: VectorStore inicializado
            config_path: Caminho para o arquivo config.toml
            llm_provider: Provedor do LLM
            context_name: Nome do contexto (ex: cond_169)

        Returns:
            Instância configurada do RAGChain
        """
        config = toml.load(config_path)

        # Configuração do LLM escolhido
        llm_config = config.get("llm", {}).get(llm_provider, {})
        retrieval_config = config.get("retrieval", {})
        prompt_config = config.get("prompt", {})

        return cls(
            vector_store=vector_store,
            llm_provider=llm_provider,
            model=llm_config.get("model"),
            temperature=llm_config.get("temperature", 0.3),
            max_tokens=llm_config.get("max_tokens", 4096),
            top_k=retrieval_config.get("top_k", 8),
            system_context=prompt_config.get("system_context", "documentos e informações disponíveis"),
            context_name=context_name,
            brand_personality=prompt_config.get("brand_personality", ""),
        )

    def _create_llm(
        self,
        provider: LLMProvider,
        model: Optional[str],
        temperature: float,
        max_tokens: int,
    ):
        """Cria instância do LLM baseado no provedor."""
        if provider == "openai":
            return ChatOpenAI(
                model=model or "gpt-4o",
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                model=model or "claude-sonnet-4-20250514",
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
        else:
            raise ValueError(f"Provedor não suportado: {provider}")

    def query(
        self,
        question: str,
        return_sources: bool = True,
    ) -> dict:
        """
        Executa query no RAG e retorna resposta.

        Args:
            question: Pergunta do usuário
            return_sources: Se True, retorna também os documentos fonte

        Returns:
            Dicionário com resposta e metadados
        """
        # 1. Recupera documentos relevantes
        documents = self.vector_store.search_documents(
            query=question,
            top_k=self.top_k,
        )

        # 2. Formata contexto em TOON
        context = self.toon_formatter.format_documents(documents)

        # 3. Gera resposta com LLM
        chain = self._prompt | self._llm | self._output_parser
        response = chain.invoke({
            "context": context,
            "question": question,
        })

        result = {
            "answer": response,
            "llm_provider": self.llm_provider,
            "context_format": self.toon_formatter.format_type,
        }

        if return_sources:
            result["sources"] = [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "file": doc.metadata.get("source", "unknown"),
                    "chunk": f"{doc.metadata.get('chunk_index', 0) + 1}/{doc.metadata.get('total_chunks', '?')}",
                }
                for doc in documents
            ]

        return result

    def query_with_history(
        self,
        question: str,
        documents: List[Document],
        conversation_history: Optional[str] = None,
        not_found_message: str = "Não encontrei informações sobre este assunto nos documentos disponíveis.",
        return_sources: bool = True,
    ) -> dict:
        """
        Executa query com histórico de conversa e documentos pré-recuperados.

        Este método é usado quando os documentos já foram recuperados de múltiplos
        contextos (hierarquia legal) e precisamos gerar a resposta com contexto
        de conversa.

        Args:
            question: Pergunta do usuário
            documents: Lista de documentos já recuperados
            conversation_history: Histórico formatado da conversa
            not_found_message: Mensagem quando não encontra resposta
            return_sources: Se True, retorna fontes

        Returns:
            Dicionário com resposta e metadados
        """
        # Formata contexto dos documentos
        context = self._format_documents_with_hierarchy(documents)

        # Prepara histórico
        history_text = ""
        if conversation_history:
            history_text = f"## Histórico da Conversa\n{conversation_history}\n---\n"

        # Usa o prompt com histórico
        prompt = ChatPromptTemplate.from_template(self._inject_personality(self.PROMPT_WITH_HISTORY_TEMPLATE))
        chain = prompt | self._llm | self._output_parser

        response = chain.invoke({
            "conversation_history": history_text,
            "context": context,
            "question": question,
            "not_found_message": not_found_message,
        })

        result = {
            "answer": response,
            "llm_provider": self.llm_provider,
            "context_format": "hierarchical",
        }

        if return_sources:
            result["sources"] = [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "file": doc.metadata.get("source", "unknown"),
                    "hierarchy_context": doc.metadata.get("hierarchy_context", "unknown"),
                    "hierarchy_name": doc.metadata.get("hierarchy_name", "Documento"),
                    "hierarchy_level": doc.metadata.get("hierarchy_level", 0),
                }
                for doc in documents
            ]

        return result

    def query_fluent(
        self,
        question: str,
        documents: List[Document],
        conversation_history: Optional[str] = None,
        show_source: bool = False,
        not_found_message: str = "Não encontrei informações sobre este assunto nos documentos disponíveis.",
        return_sources: bool = True,
    ) -> dict:
        """
        Executa query com resposta FLUIDA (sem mencionar hierarquia técnica).

        Este é o método recomendado para respostas ao usuário final.
        A resposta é clara e direta, sem termos técnicos.

        Args:
            question: Pergunta do usuário
            documents: Lista de documentos já recuperados
            conversation_history: Histórico formatado da conversa
            show_source: Se True, menciona a fonte NA RESPOSTA (ex: "De acordo com...")
            not_found_message: Mensagem quando não encontra resposta
            return_sources: Se True, retorna fontes no JSON

        Returns:
            Dicionário com resposta e metadados
        """
        if not documents:
            return {
                "answer": not_found_message,
                "llm_provider": self.llm_provider,
                "context_format": "fluent",
                "sources": [] if return_sources else None,
            }

        # Formata contexto dos documentos (simplificado para resposta fluida)
        context = self._format_documents_simple(documents)

        # Prepara histórico
        history_text = ""
        if conversation_history:
            history_text = f"Contexto da conversa anterior:\n{conversation_history}\n---\n"

        # Escolhe prompt baseado em show_source
        if show_source:
            prompt_template = self.FLUENT_WITH_SOURCE_PROMPT_TEMPLATE
        else:
            prompt_template = self.FLUENT_PROMPT_TEMPLATE

        prompt = ChatPromptTemplate.from_template(self._inject_personality(prompt_template))
        chain = prompt | self._llm | self._output_parser

        response = chain.invoke({
            "conversation_history": history_text,
            "context": context,
            "question": question,
            "not_found_message": not_found_message,
        })

        result = {
            "answer": response,
            "llm_provider": self.llm_provider,
            "context_format": "fluent",
            "show_source": show_source,
        }

        if return_sources:
            result["sources"] = [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "file": doc.metadata.get("source", "unknown"),
                    "hierarchy_context": doc.metadata.get("hierarchy_context", "unknown"),
                    "hierarchy_name": doc.metadata.get("hierarchy_name", "Documento"),
                    "hierarchy_level": doc.metadata.get("hierarchy_level", 0),
                }
                for doc in documents
            ]

        return result

    def _format_documents_simple(self, documents: List[Document]) -> str:
        """
        Formata documentos de forma simples (sem marcadores de hierarquia).

        Args:
            documents: Lista de documentos

        Returns:
            String formatada com conteúdo dos documentos
        """
        if not documents:
            return "Nenhum documento encontrado."

        formatted_parts = []
        for doc in documents:
            source = doc.metadata.get("source", "documento")
            formatted_parts.append(f"[{source}]\n{doc.page_content}")

        return "\n\n---\n\n".join(formatted_parts)

    def _format_documents_with_hierarchy(self, documents: List[Document]) -> str:
        """
        Formata documentos com indicação de hierarquia.

        Args:
            documents: Lista de documentos com metadados de hierarquia

        Returns:
            String formatada com indicação de nível hierárquico
        """
        if not documents:
            return "Nenhum documento relevante encontrado."

        # Agrupa por nível hierárquico
        by_level: Dict[int, List[Document]] = {}
        for doc in documents:
            level = doc.metadata.get("hierarchy_level", 99)
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(doc)

        # Formata em ordem de hierarquia
        formatted_parts = []
        for level in sorted(by_level.keys()):
            docs = by_level[level]
            if docs:
                hierarchy_name = docs[0].metadata.get("hierarchy_name", f"Nível {level}")

                # Emoji baseado no nível
                emoji = "⚖️" if level <= 2 else "📄"
                formatted_parts.append(f"\n{emoji} **{hierarchy_name}**:")

                for doc in docs:
                    source = doc.metadata.get("source", "unknown")
                    formatted_parts.append(f"  [{source}] {doc.page_content}")

        return "\n".join(formatted_parts)

    def query_with_scores(
        self,
        question: str,
    ) -> dict:
        """
        Executa query retornando também scores de relevância.

        Args:
            question: Pergunta do usuário

        Returns:
            Dicionário com resposta, fontes e scores
        """
        # Recupera com scores
        results = self.vector_store.search(
            query=question,
            top_k=self.top_k,
        )

        documents = [doc for doc, _ in results]

        # Formata contexto
        context = self.toon_formatter.format_with_scores(results)

        # Gera resposta
        chain = self._prompt | self._llm | self._output_parser
        response = chain.invoke({
            "context": context,
            "question": question,
        })

        return {
            "answer": response,
            "sources": [
                {
                    "content": doc.page_content[:200] + "...",
                    "file": doc.metadata.get("source", "unknown"),
                    "relevance": round(1 - score, 3),
                }
                for doc, score in results
            ],
            "llm_provider": self.llm_provider,
        }

    def switch_llm(
        self,
        provider: LLMProvider,
        model: Optional[str] = None,
    ) -> None:
        """
        Troca o LLM em uso.

        Args:
            provider: Novo provedor ("openai" ou "anthropic")
            model: Novo modelo (opcional)
        """
        self.llm_provider = provider
        self._llm = self._create_llm(
            provider=provider,
            model=model,
            temperature=0.3,
            max_tokens=4096,
        )

    def get_info(self) -> dict:
        """Retorna informações sobre a configuração atual."""
        return {
            "llm_provider": self.llm_provider,
            "model": self._llm.model_name if hasattr(self._llm, 'model_name') else str(self._llm.model),
            "top_k": self.top_k,
            "context_format": self.toon_formatter.format_type,
            "vector_store_initialized": self.vector_store.is_initialized,
        }
