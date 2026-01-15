"""RAG Chain - Pipeline principal de Retrieval-Augmented Generation."""

import os
from typing import List, Optional, Literal

import toml
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from .vector_store import VectorStore
from .toon_formatter import ToonFormatter


LLMProvider = Literal["openai", "anthropic"]


class RAGChain:
    """Pipeline completo de RAG com suporte a múltiplos LLMs."""

    DEFAULT_PROMPT_TEMPLATE = """Você é um assistente jurídico especializado em direito condominial brasileiro.

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

        # Configura LLM
        self._llm = self._create_llm(
            provider=llm_provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Configura prompt com o contexto do sistema
        prompt_text = self.DEFAULT_PROMPT_TEMPLATE.replace("{system_context}", system_context)
        self._prompt = ChatPromptTemplate.from_template(prompt_text)
        self._output_parser = StrOutputParser()

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
