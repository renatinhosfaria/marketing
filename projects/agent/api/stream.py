"""
Utilidades de streaming SSE.

sse_event(): formata um evento SSE padrao.
_safe_json(): tenta parsear content como JSON, retorna string se falhar.
"""

import json


def sse_event(event_type: str, data: dict) -> str:
    """Formata evento SSE padrao.

    Args:
        event_type: Tipo do evento (message, agent_status, tool_result, interrupt, done).
        data: Dados do evento (dict).

    Returns:
        String formatada como evento SSE.
    """
    # ensure_ascii=False: PT-BR bonito no stream (acentos preservados)
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def sse_event_with_id(event_id: str, event_type: str, data: dict) -> str:
    """Formata evento SSE com id para suporte a Last-Event-ID.

    Args:
        event_id: ID do evento (Redis Stream ID).
        event_type: Tipo do evento.
        data: Dados do evento (dict).

    Returns:
        String formatada como evento SSE com campo id.
    """
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"id:{event_id}\nevent:{event_type}\ndata:{payload}\n\n"


def _safe_json(content: str) -> dict | str:
    """Tenta parsear content como JSON; retorna string se falhar.

    Args:
        content: String potencialmente JSON.

    Returns:
        dict se parsing bem-sucedido, string original caso contrario.
    """
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content
