"""
Client HTTP persistente para chamadas a ML API.

Combina:
  - Rate limiting por account (Semaphore)
  - Error handling padronizado (ToolResult)
  - Pool de conexoes (httpx.AsyncClient keep-alive)
  - Timeout configuravel

Funcoes:
  - init_ml_http_client(): cria client persistente (chamado no lifespan)
  - close_ml_http_client(): fecha client (chamado no shutdown)
  - _ml_api_call(): helper para chamadas com rate limiting + error handling
"""

import asyncio
import time
from collections import defaultdict

import httpx

from projects.agent.config import agent_settings
from projects.agent.tools.result import ToolResult, tool_success, tool_error
from projects.agent.observability.metrics import ml_api_latency, tool_errors_total


# Semaphores por account — limita concorrencia real
_ml_api_semaphores: dict[str, asyncio.Semaphore] = defaultdict(
    lambda: asyncio.Semaphore(5)  # Max 5 chamadas ML API simultaneas por account
)
_fb_api_semaphores: dict[str, asyncio.Semaphore] = defaultdict(
    lambda: asyncio.Semaphore(3)  # Max 3 chamadas Facebook API simultaneas por account
)

# Client HTTP persistente — inicializado no lifespan da app.
# Reutiliza pool de conexoes (keep-alive HTTP/1.1).
_ml_http_client: httpx.AsyncClient | None = None

ML_API_URL = agent_settings.ml_api_url


def init_ml_http_client() -> httpx.AsyncClient:
    """Cria o client HTTP persistente. Chamado no lifespan da app."""
    global _ml_http_client
    _ml_http_client = httpx.AsyncClient(
        base_url=ML_API_URL,
        timeout=float(agent_settings.ml_api_timeout),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    return _ml_http_client


async def close_ml_http_client():
    """Fecha o client no shutdown. Chamado no lifespan da app."""
    global _ml_http_client
    if _ml_http_client:
        await _ml_http_client.aclose()
        _ml_http_client = None


async def _ml_api_call(
    method: str,
    path: str,
    *,
    account_id: str = None,
    **kwargs,
) -> ToolResult:
    """Helper para chamadas a ML API com rate limiting + error handling.

    Retorna ToolResult padronizado.
    Usa client HTTP persistente (pool de conexoes).

    Args:
        method: Metodo HTTP (get, post, put, delete).
        path: Path relativo ao base_url.
        account_id: ID da conta para rate limiting.
        **kwargs: Argumentos adicionais para httpx (json, params, headers, timeout).
    """
    timeout = kwargs.pop("timeout", agent_settings.ml_api_timeout)
    headers = kwargs.pop("headers", {}) or {}
    headers = {"X-Agent-Version": agent_settings.agent_version, **headers}

    sem = _ml_api_semaphores[account_id or "global"]
    client = _ml_http_client

    if client is None:
        return tool_error(
            "UNAVAILABLE",
            "ML HTTP client nao inicializado. Verifique o lifespan da app.",
        )

    async with sem:
        start = time.monotonic()
        try:
            resp = await getattr(client, method)(
                path,
                timeout=float(timeout),
                headers=headers,
                **kwargs,
            )
            resp.raise_for_status()
            ml_api_latency.labels(
                method=method, path=path, status=str(resp.status_code),
            ).observe(time.monotonic() - start)
            return tool_success(resp.json())
        except httpx.TimeoutException:
            ml_api_latency.labels(
                method=method, path=path, status="timeout",
            ).observe(time.monotonic() - start)
            tool_errors_total.labels(
                tool_name="ml_api_call", error_code="TIMEOUT",
            ).inc()
            return tool_error(
                "TIMEOUT",
                f"ML API timeout em {path}. Tente novamente em instantes.",
                retryable=True,
            )
        except httpx.HTTPStatusError as e:
            ml_api_latency.labels(
                method=method, path=path, status=str(e.response.status_code),
            ).observe(time.monotonic() - start)
            tool_errors_total.labels(
                tool_name="ml_api_call", error_code="HTTP_ERROR",
            ).inc()
            return tool_error(
                "HTTP_ERROR",
                f"ML API retornou {e.response.status_code} em {path}.",
                retryable=e.response.status_code >= 500,
            )
        except httpx.ConnectError:
            ml_api_latency.labels(
                method=method, path=path, status="connect_error",
            ).observe(time.monotonic() - start)
            tool_errors_total.labels(
                tool_name="ml_api_call", error_code="UNAVAILABLE",
            ).inc()
            return tool_error(
                "UNAVAILABLE",
                "ML API indisponivel. Verifique se o servico esta rodando.",
                retryable=True,
            )
