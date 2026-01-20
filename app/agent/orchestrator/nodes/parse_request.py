"""Nó parse_request do Orchestrator.

Responsável por analisar a mensagem do usuário e detectar a intenção.
"""
import re
from typing import Optional, TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, BaseMessage

if TYPE_CHECKING:
    from app.agent.orchestrator.state import OrchestratorState


# Padrões de intenção (regex case-insensitive)
INTENT_PATTERNS: dict[str, list[str]] = {
    "analyze_performance": [
        r"como\s+est[áa]",
        r"analis[ea]",
        r"performance",
        r"resultado",
        r"m[ée]trica",
        r"desempenho",
        r"status\s+da",
    ],
    "find_problems": [
        r"problema",
        r"anomalia",
        r"errado",
        r"issue",
        r"alert",
        r"cr[íi]tic",
    ],
    "get_recommendations": [
        r"recomenda",
        r"sugest[ãa]o",
        r"o\s+que\s+(fazer|devo)",
        r"escalar",
        r"otimizar",
        r"melhorar",
        r"a[çc][ãa]o",
    ],
    "predict_future": [
        r"previs[ãa]o",
        r"forecast",
        r"prever",
        r"futuro",
        r"pr[óo]xim[oa]",
        r"vai\s+ser",
    ],
    "compare_campaigns": [
        r"compar[ae]",
        r"versus",
        r"\s+vs\s+",
        r"melhor\s+entre",
    ],
    "full_report": [
        r"relat[óo]rio\s+completo",
        r"resumo\s+geral",
        r"vis[ãa]o\s+geral",
        r"tudo\s+sobre",
    ],
}


def detect_intent(message: str) -> str:
    """Detecta a intenção do usuário baseado na mensagem.

    Args:
        message: Mensagem do usuário

    Returns:
        String com a intenção detectada (ou "general" como fallback)
    """
    message_lower = message.lower()
    scores: dict[str, int] = {intent: 0 for intent in INTENT_PATTERNS}

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, message_lower):
                scores[intent] += 1

    max_score = max(scores.values())
    if max_score == 0:
        return "general"

    # Retorna o primeiro intent com score máximo
    for intent, score in scores.items():
        if score == max_score:
            return intent

    return "general"


def extract_campaign_references(message: str) -> list[str]:
    """Extrai referências a campanhas da mensagem.

    Args:
        message: Mensagem do usuário

    Returns:
        Lista de nomes de campanhas mencionadas
    """
    campaigns: list[str] = []

    # Padrão: "campanha 'X'" ou 'campanha "X"' (com aspas)
    pattern_quoted = r"campanha\s+['\"]([^'\"]+)['\"]"
    matches_quoted = re.findall(pattern_quoted, message, re.IGNORECASE)
    campaigns.extend([m.strip() for m in matches_quoted])

    # Padrão: "campanha X" sem aspas - captura até pontuação ou final
    # Captura palavras depois de "campanha " até encontrar ? . , ou fim da string
    pattern_unquoted = r"campanha\s+([A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9\s]+?)(?:[?.,]|$)"
    matches_unquoted = re.findall(pattern_unquoted, message, re.IGNORECASE)
    campaigns.extend([m.strip() for m in matches_unquoted])

    # Padrão: texto entre aspas (simples ou duplas) - genérico
    quoted = re.findall(r'["\']([^"\']+)["\']', message)
    campaigns.extend(quoted)

    # Remove duplicatas mantendo ordem
    seen: set[str] = set()
    unique_campaigns: list[str] = []
    for c in campaigns:
        if c not in seen:
            seen.add(c)
            unique_campaigns.append(c)

    return unique_campaigns


def parse_request(state: "OrchestratorState") -> dict:
    """Nó que analisa a requisição do usuário.

    Extrai a intenção e referências a campanhas da última mensagem
    do usuário no estado.

    Args:
        state: Estado atual do orchestrator

    Returns:
        Dicionário com intent, campaign_refs e original_message
    """
    messages = state.get("messages", [])
    user_message: Optional[str] = None

    # Encontra a última mensagem do usuário (percorre de trás pra frente)
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    if not user_message:
        return {"intent": "general", "campaign_refs": []}

    intent = detect_intent(user_message)
    campaign_refs = extract_campaign_references(user_message)

    return {
        "intent": intent,
        "campaign_refs": campaign_refs,
        "original_message": user_message,
    }
