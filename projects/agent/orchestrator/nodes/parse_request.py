"""Nó parse_request do Orchestrator.

Responsável por analisar a mensagem do usuário e detectar a intenção.
"""
import re
from typing import Optional

from langchain_core.messages import HumanMessage

from projects.agent.orchestrator.state import OrchestratorState
from shared.core.logging import get_logger
from shared.core.tracing.decorators import log_span
from shared.core.tracing.events import log_intent_detected

logger = get_logger("orchestrator.parse_request")


# Padrões de intenção (regex case-insensitive)
INTENT_PATTERNS = {
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
        r"troubleshoot",
        r"diagn[óo]stic",
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
        r"tend[êe]ncia",
        r"vai\s+ser",
        r"estima",
    ],
    "compare_campaigns": [
        r"compar[ae]",
        r"versus",
        r"\s+vs\s+",
        r"melhor\s+entre",
        r"diferen[çc]a",
        r"lado\s+a\s+lado",
    ],
    "full_report": [
        r"relat[óo]rio\s+completo",
        r"resumo\s+geral",
        r"vis[ãa]o\s+geral",
        r"overview",
        r"tudo\s+sobre",
        r"an[áa]lise\s+completa",
    ],
    "troubleshoot": [
        r"por\s+que",
        r"motivo",
        r"causa",
        r"investig",
        r"debug",
        r"entender",
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
    campaigns = []

    # Padrão: "campanha X" ou "campanha 'X'"
    pattern = r"campanha\s+['\"]?([^'\"]+)['\"]?"
    matches = re.findall(pattern, message, re.IGNORECASE)
    campaigns.extend(matches)

    # Padrão: menção direta com aspas
    quoted = re.findall(r'["\']([^"\']+)["\']', message)
    campaigns.extend(quoted)

    return list(set(campaigns))


@log_span("intent_detection", log_args=True, log_result=False)
def parse_request(state: OrchestratorState) -> dict:
    """Nó que analisa a requisição do usuário.

    Extrai a intenção e referências a campanhas da última mensagem
    do usuário no estado.

    Args:
        state: Estado atual do orchestrator

    Returns:
        Dicionário com intent, campaign_refs e original_message
    """
    logger.info("Analisando requisicao do usuario")

    # Obter ultima mensagem do usuario
    messages = state.get("messages", [])
    user_message = None

    # Encontra a última mensagem do usuário (percorre de trás pra frente)
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    if not user_message:
        logger.warning("Nenhuma mensagem de usuario encontrada")
        return {
            "user_intent": "general",
            "error": "Nenhuma mensagem encontrada"
        }

    # Detectar intencao
    intent = detect_intent(user_message)
    logger.info("Intencao detectada: {intent}")

    # Contar quantos patterns fizeram match para logging
    matched_patterns_count = 0
    if intent in INTENT_PATTERNS:
        for pattern in INTENT_PATTERNS[intent]:
            if re.search(pattern, user_message.lower()):
                matched_patterns_count += 1

    # Logar intent detectado
    log_intent_detected(
        intent=intent,
        confidence=1.0,  # Regex-based detection has 100% confidence in pattern match
        reasoning=f"Detected via regex patterns: {matched_patterns_count} patterns matched"
    )

    # Extrair campanhas mencionadas
    campaigns = extract_campaign_references(user_message)
    if campaigns:
        logger.debug("Campanhas mencionadas: {campaigns}")

    return {
        "user_intent": intent,
    }
