"""
Conversation Manager - Gerenciamento de histórico de conversas.

Este módulo integra com o chatbot_new para:
- Buscar histórico de conversas por conversation_id
- Formatar histórico para inclusão no prompt do LLM
- Gerenciar contexto de conversa em memória
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Representa uma mensagem na conversa."""
    role: str  # "user" ou "assistant"
    content: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Cria Message a partir de dicionário."""
        timestamp = None
        if data.get("timestamp"):
            try:
                timestamp = datetime.fromisoformat(data["timestamp"])
            except (ValueError, TypeError):
                pass

        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=timestamp,
            metadata=data.get("metadata"),
        )


class ConversationManager:
    """
    Gerencia histórico de conversas e integração com chatbot externo.

    Responsabilidades:
    - Buscar histórico de conversas do chatbot_new via API
    - Manter cache local de conversas ativas
    - Formatar histórico para inclusão no prompt do LLM
    - Limitar número de mensagens no contexto
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        max_history: int = 10,
        enabled: bool = True,
        timeout: float = 5.0,
    ):
        """
        Inicializa o ConversationManager.

        Args:
            api_url: URL da API do chatbot_new para buscar histórico
            max_history: Máximo de mensagens a incluir no contexto
            enabled: Se o gerenciamento de histórico está ativado
            timeout: Timeout para requisições HTTP
        """
        self.api_url = api_url
        self.max_history = max_history
        self.enabled = enabled
        self.timeout = timeout

        # Cache local de conversas
        self._cache: Dict[str, List[Message]] = {}

        logger.info(
            f"✅ ConversationManager inicializado "
            f"(enabled={enabled}, max_history={max_history}, api_url={api_url})"
        )

    @classmethod
    def from_governance(cls, governance) -> "ConversationManager":
        """
        Cria ConversationManager a partir de GovernanceManager.

        Args:
            governance: Instância do GovernanceManager

        Returns:
            ConversationManager configurado
        """
        conv_config = governance.get_conversation_config()
        return cls(
            api_url=conv_config.get("api_url"),
            max_history=conv_config.get("max_history", 10),
            enabled=conv_config.get("enabled", True),
        )

    async def fetch_conversation_history(
        self,
        conversation_id: str,
    ) -> List[Message]:
        """
        Busca histórico de conversa do chatbot_new.

        Args:
            conversation_id: ID da conversa no chatbot_new

        Returns:
            Lista de mensagens da conversa
        """
        if not self.enabled or not self.api_url or not conversation_id:
            return []

        # Verifica cache primeiro
        if conversation_id in self._cache:
            logger.debug(f"📋 Histórico do cache: {conversation_id}")
            return self._cache[conversation_id][-self.max_history:]

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Endpoint esperado: GET /api/conversations/{id}/messages
                url = f"{self.api_url}/{conversation_id}/messages"
                logger.info(f"🔄 Buscando histórico: {url}")

                response = await client.get(url)
                response.raise_for_status()

                data = response.json()
                messages_data = data.get("messages", data) if isinstance(data, dict) else data

                # Converte para objetos Message
                messages = [Message.from_dict(msg) for msg in messages_data]

                # Armazena no cache
                self._cache[conversation_id] = messages

                logger.info(f"✅ Histórico carregado: {len(messages)} mensagens")
                return messages[-self.max_history:]

        except httpx.TimeoutException:
            logger.warning(f"⚠️ Timeout ao buscar histórico: {conversation_id}")
            return []
        except httpx.HTTPStatusError as e:
            logger.warning(f"⚠️ Erro HTTP ao buscar histórico: {e.response.status_code}")
            return []
        except Exception as e:
            logger.warning(f"⚠️ Erro ao buscar histórico: {e}")
            return []

    def fetch_conversation_history_sync(
        self,
        conversation_id: str,
    ) -> List[Message]:
        """
        Versão síncrona de fetch_conversation_history.

        Args:
            conversation_id: ID da conversa

        Returns:
            Lista de mensagens
        """
        if not self.enabled or not self.api_url or not conversation_id:
            return []

        # Verifica cache primeiro
        if conversation_id in self._cache:
            return self._cache[conversation_id][-self.max_history:]

        try:
            with httpx.Client(timeout=self.timeout) as client:
                url = f"{self.api_url}/{conversation_id}/messages"
                response = client.get(url)
                response.raise_for_status()

                data = response.json()
                messages_data = data.get("messages", data) if isinstance(data, dict) else data

                messages = [Message.from_dict(msg) for msg in messages_data]
                self._cache[conversation_id] = messages

                return messages[-self.max_history:]

        except Exception as e:
            logger.warning(f"⚠️ Erro ao buscar histórico (sync): {e}")
            return []

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Adiciona mensagem ao cache local.

        Args:
            conversation_id: ID da conversa
            role: "user" ou "assistant"
            content: Conteúdo da mensagem
            metadata: Metadados adicionais
        """
        if not conversation_id:
            return

        if conversation_id not in self._cache:
            self._cache[conversation_id] = []

        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata,
        )
        self._cache[conversation_id].append(message)

        # Limita tamanho do cache
        if len(self._cache[conversation_id]) > self.max_history * 2:
            self._cache[conversation_id] = self._cache[conversation_id][-self.max_history:]

    def get_cached_history(
        self,
        conversation_id: str,
    ) -> List[Message]:
        """
        Retorna histórico do cache local.

        Args:
            conversation_id: ID da conversa

        Returns:
            Lista de mensagens em cache
        """
        return self._cache.get(conversation_id, [])[-self.max_history:]

    def clear_cache(self, conversation_id: Optional[str] = None) -> None:
        """
        Limpa cache de conversas.

        Args:
            conversation_id: Se especificado, limpa apenas esta conversa
        """
        if conversation_id:
            self._cache.pop(conversation_id, None)
        else:
            self._cache.clear()

    def format_history_for_prompt(
        self,
        messages: List[Message],
        include_metadata: bool = False,
    ) -> str:
        """
        Formata histórico para inclusão no prompt do LLM.

        Args:
            messages: Lista de mensagens
            include_metadata: Se deve incluir metadados

        Returns:
            String formatada com histórico
        """
        if not messages:
            return ""

        formatted_parts = ["## Histórico da Conversa\n"]

        for msg in messages:
            role_label = "USUÁRIO" if msg.role == "user" else "ASSISTENTE"
            formatted_parts.append(f"**{role_label}**: {msg.content}")

            if include_metadata and msg.metadata:
                meta_str = ", ".join(f"{k}={v}" for k, v in msg.metadata.items())
                formatted_parts.append(f"  _({meta_str})_")

        formatted_parts.append("\n---\n")

        return "\n".join(formatted_parts)

    def format_messages_for_llm(
        self,
        messages: List[Message],
    ) -> List[Dict[str, str]]:
        """
        Formata mensagens no formato esperado pelo LLM (OpenAI/Anthropic).

        Args:
            messages: Lista de mensagens

        Returns:
            Lista de dicts com role/content
        """
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

    def merge_histories(
        self,
        api_history: List[Message],
        request_history: Optional[List[Dict]] = None,
    ) -> List[Message]:
        """
        Combina histórico da API com histórico do request.

        Prioriza o request_history se fornecido (mais recente).

        Args:
            api_history: Histórico buscado da API
            request_history: Histórico enviado no request (opcional)

        Returns:
            Lista combinada de mensagens
        """
        if request_history:
            # Request history tem prioridade (mais atualizado)
            return [Message.from_dict(msg) for msg in request_history][-self.max_history:]

        return api_history[-self.max_history:]

    def get_context_window_info(self, messages: List[Message]) -> Dict[str, Any]:
        """
        Retorna informações sobre a janela de contexto.

        Args:
            messages: Lista de mensagens

        Returns:
            Dict com estatísticas
        """
        total_chars = sum(len(msg.content) for msg in messages)
        user_messages = sum(1 for msg in messages if msg.role == "user")
        assistant_messages = sum(1 for msg in messages if msg.role == "assistant")

        return {
            "total_messages": len(messages),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "total_characters": total_chars,
            "estimated_tokens": total_chars // 4,  # Estimativa aproximada
        }
