"""
Governance Manager - Gerenciamento de hierarquia legal e perfis de acesso.

Este módulo controla:
- Hierarquia de documentos (Código Civil > Lei 4591 > Convenção > Regimento > Atas)
- Perfis de usuário (anônimo, autenticado, admin)
- Contextos permitidos por perfil
- Mensagens de fallback quando não encontra resposta
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import toml

logger = logging.getLogger(__name__)


class UserProfile(Enum):
    """Perfis de acesso do sistema."""
    ANONYMOUS = "anonymous"
    AUTHENTICATED = "authenticated"
    ADMIN = "admin"


@dataclass
class ProfileConfig:
    """Configuração de um perfil de acesso."""
    contexts: List[str] = field(default_factory=list)
    max_sources: int = 5
    response_template: str = "detailed"
    can_access_private: bool = False


@dataclass
class GovernanceRules:
    """Regras de governança carregadas do config."""
    # Hierarquia de documentos (ordem de precedência legal)
    document_hierarchy: List[str] = field(default_factory=lambda: [
        "codigo_civil",
        "lei_condominios",
        "convencao",
        "regimento_interno",
        "ata_assembleia",
        "avisos"
    ])

    # Contextos de legislação federal (sempre consultados primeiro)
    legal_contexts: List[str] = field(default_factory=lambda: [
        "codigo_civil",
        "lei_condominios"
    ])

    # Perfis de acesso
    profiles: Dict[str, ProfileConfig] = field(default_factory=dict)

    # Mensagens
    not_found_message: str = "Não encontrei informações sobre este assunto nos documentos disponíveis."
    hierarchy_not_found_message: str = "Não encontrei na {requested_level}. {fallback_message}"

    # Comportamento de fallback
    include_fallback_sources: bool = True
    include_fallback_notice: bool = True

    # Mapeamento de contextos por autenticação
    context_mapping: Dict[str, str] = field(default_factory=dict)

    # Configuração de conversa
    conversation_enabled: bool = True
    max_history_messages: int = 10
    conversation_api_url: Optional[str] = None


class GovernanceManager:
    """
    Gerencia regras de governança, hierarquia e perfis de acesso.

    Responsabilidades:
    - Determinar contextos permitidos por perfil
    - Ordenar busca por hierarquia legal
    - Gerar mensagens de not_found apropriadas
    - Controlar acesso a documentos privados/públicos
    """

    def __init__(self, config_path: str = "config.toml"):
        """
        Inicializa o GovernanceManager.

        Args:
            config_path: Caminho para o arquivo config.toml
        """
        self.config_path = config_path
        self.rules = self._load_rules(config_path)
        logger.info(f"✅ GovernanceManager inicializado com {len(self.rules.profiles)} perfis")

    def _load_rules(self, config_path: str) -> GovernanceRules:
        """Carrega regras do arquivo de configuração."""
        rules = GovernanceRules()

        try:
            config = toml.load(config_path)
            governance_config = config.get("governance", {})

            # Hierarquia de documentos
            if "document_hierarchy" in governance_config:
                rules.document_hierarchy = governance_config["document_hierarchy"]

            # Contextos legais (federais)
            legal_hierarchy = config.get("legal_hierarchy", {})
            rules.legal_contexts = [
                legal_hierarchy.get("codigo_civil_context", "codigo_civil"),
                legal_hierarchy.get("lei_condominios_context", "lei_condominios"),
            ]

            # Perfis de acesso
            profiles_config = governance_config.get("profiles", {})
            for profile_name, profile_data in profiles_config.items():
                rules.profiles[profile_name] = ProfileConfig(
                    contexts=profile_data.get("contexts", []),
                    max_sources=profile_data.get("max_sources", 5),
                    response_template=profile_data.get("response_template", "detailed"),
                    can_access_private=profile_data.get("can_access_private", False),
                )

            # Adiciona perfis padrão se não existirem
            if "anonymous" not in rules.profiles:
                rules.profiles["anonymous"] = ProfileConfig(
                    contexts=["*"],  # Acesso a todos por padrão
                    max_sources=3,
                    response_template="short",
                )
            if "authenticated" not in rules.profiles:
                rules.profiles["authenticated"] = ProfileConfig(
                    contexts=["*"],
                    max_sources=8,
                    response_template="detailed",
                )
            if "admin" not in rules.profiles:
                rules.profiles["admin"] = ProfileConfig(
                    contexts=["*"],
                    max_sources=20,
                    response_template="full",
                    can_access_private=True,
                )

            # Regras de resposta
            rules_config = governance_config.get("rules", {})
            rules.not_found_message = rules_config.get(
                "not_found_message",
                "Não encontrei informações sobre este assunto nos documentos disponíveis."
            )
            rules.hierarchy_not_found_message = rules_config.get(
                "hierarchy_not_found_message",
                "Não encontrei na {requested_level}. {fallback_message}"
            )
            rules.include_fallback_sources = rules_config.get("include_fallback_sources", True)
            rules.include_fallback_notice = rules_config.get("include_fallback_notice", True)

            # Mapeamento de contextos
            rules.context_mapping = governance_config.get("context_mapping", {})

            # Configuração de conversa
            conv_config = governance_config.get("conversation", {})
            rules.conversation_enabled = conv_config.get("enable_history", True)
            rules.max_history_messages = conv_config.get("max_history_messages", 10)
            rules.conversation_api_url = conv_config.get("fetch_from_api")

            logger.info(f"📋 Hierarquia: {rules.document_hierarchy}")
            logger.info(f"👥 Perfis: {list(rules.profiles.keys())}")

        except FileNotFoundError:
            logger.warning(f"⚠️ Config não encontrado: {config_path}, usando padrões")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar config: {e}, usando padrões")

        return rules

    def get_user_profile(
        self,
        user_id: Optional[str] = None,
        is_authenticated: bool = False,
        is_admin: bool = False,
    ) -> UserProfile:
        """
        Determina o perfil do usuário.

        Args:
            user_id: ID do usuário (opcional)
            is_authenticated: Se está autenticado
            is_admin: Se é administrador

        Returns:
            UserProfile correspondente
        """
        if is_admin or user_id == "admin":
            return UserProfile.ADMIN
        elif is_authenticated or user_id:
            return UserProfile.AUTHENTICATED
        else:
            return UserProfile.ANONYMOUS

    def get_allowed_contexts(
        self,
        requested_context: str,
        profile: UserProfile,
    ) -> List[str]:
        """
        Retorna contextos permitidos para o perfil.

        Args:
            requested_context: Contexto solicitado (ex: cond_0388)
            profile: Perfil do usuário

        Returns:
            Lista de contextos permitidos (pode incluir legislação)
        """
        profile_config = self.rules.profiles.get(profile.value, ProfileConfig())
        allowed = profile_config.contexts

        # "*" significa acesso a tudo
        if "*" in allowed:
            # Retorna contextos legais + contexto solicitado
            return self.rules.legal_contexts + [requested_context]

        # Verifica se contexto solicitado está permitido
        if requested_context in allowed:
            return self.rules.legal_contexts + [requested_context]

        # Verifica mapeamento de contexto
        mapped = self.rules.context_mapping.get(requested_context)
        if mapped and mapped in allowed:
            return self.rules.legal_contexts + [mapped]

        # Se não tem permissão, retorna apenas contextos legais públicos
        logger.warning(f"⚠️ Acesso negado ao contexto '{requested_context}' para perfil '{profile.value}'")
        return self.rules.legal_contexts

    def get_document_hierarchy(
        self,
        requested_level: Optional[str] = None,
    ) -> List[str]:
        """
        Retorna a hierarquia de documentos para busca.

        Args:
            requested_level: Nível específico solicitado (ex: "convencao")

        Returns:
            Lista ordenada de níveis hierárquicos
        """
        hierarchy = self.rules.document_hierarchy.copy()

        if requested_level:
            # Se pediu nível específico, coloca na frente
            requested_lower = requested_level.lower().replace(" ", "_")

            # Tenta encontrar o nível na hierarquia
            for i, level in enumerate(hierarchy):
                if level.lower() == requested_lower or requested_lower in level.lower():
                    # Move para o início
                    hierarchy.insert(0, hierarchy.pop(i))
                    break

        return hierarchy

    def get_hierarchy_contexts(
        self,
        condo_context: str,
        include_legal: bool = True,
    ) -> List[str]:
        """
        Retorna contextos na ordem hierárquica para busca.

        Args:
            condo_context: Contexto do condomínio (ex: cond_0388)
            include_legal: Se deve incluir legislação federal

        Returns:
            Lista ordenada: [codigo_civil, lei_condominios, condo_context]
        """
        contexts = []

        if include_legal:
            contexts.extend(self.rules.legal_contexts)

        if condo_context and condo_context not in contexts:
            contexts.append(condo_context)

        return contexts

    def get_not_found_message(
        self,
        requested_level: Optional[str] = None,
    ) -> str:
        """
        Retorna mensagem apropriada quando não encontra resposta.

        Args:
            requested_level: Nível específico que foi solicitado

        Returns:
            Mensagem formatada
        """
        if requested_level:
            return self.rules.hierarchy_not_found_message.format(
                requested_level=requested_level,
                fallback_message=self.rules.not_found_message,
            )

        return self.rules.not_found_message

    def should_include_fallback(self) -> bool:
        """Se deve mostrar fontes de outros níveis quando não encontrou no pedido."""
        return self.rules.include_fallback_sources

    def should_include_fallback_notice(self) -> bool:
        """Se deve avisar quando usa fallback de outro nível."""
        return self.rules.include_fallback_notice

    def get_max_sources(self, profile: UserProfile) -> int:
        """Retorna número máximo de fontes para o perfil."""
        profile_config = self.rules.profiles.get(profile.value, ProfileConfig())
        return profile_config.max_sources

    def get_response_template(self, profile: UserProfile) -> str:
        """Retorna template de resposta para o perfil."""
        profile_config = self.rules.profiles.get(profile.value, ProfileConfig())
        return profile_config.response_template

    def get_conversation_config(self) -> Dict[str, Any]:
        """Retorna configuração de conversa."""
        return {
            "enabled": self.rules.conversation_enabled,
            "max_history": self.rules.max_history_messages,
            "api_url": self.rules.conversation_api_url,
        }

    def get_hierarchy_level_name(self, level: str) -> str:
        """
        Retorna nome amigável do nível hierárquico.

        Args:
            level: Identificador do nível (ex: "codigo_civil")

        Returns:
            Nome legível (ex: "Código Civil")
        """
        names = {
            "codigo_civil": "Código Civil (Lei 10.406/2002)",
            "lei_condominios": "Lei de Condomínios (Lei 4.591/64)",
            "convencao": "Convenção do Condomínio",
            "regimento_interno": "Regimento Interno",
            "ata_assembleia": "Ata de Assembleia",
            "avisos": "Avisos e Comunicados",
        }
        return names.get(level, level.replace("_", " ").title())
