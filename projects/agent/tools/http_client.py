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
from typing import TYPE_CHECKING

import httpx

from projects.agent.config import agent_settings
from projects.agent.tools.result import ToolResult, tool_success, tool_error
from projects.agent.observability.metrics import ml_api_latency, tool_errors_total

if TYPE_CHECKING:
    from projects.agent.concurrency.redis_semaphore import RedisSemaphoreFactory

# Semaphores in-memory: fallback quando Redis nao esta disponivel (testes/dev)
_ml_api_semaphores: dict[str, asyncio.Semaphore] = defaultdict(
    lambda: asyncio.Semaphore(5)  # Max 5 chamadas ML API simultaneas por account
)

# Factory de semaphores Redis — injetada no lifespan via set_semaphore_factory()
_sem_factory: "RedisSemaphoreFactory | None" = None

# Client HTTP persistente — inicializado no lifespan da app.
# Reutiliza pool de conexoes (keep-alive HTTP/1.1).
_ml_http_client: httpx.AsyncClient | None = None

ML_API_URL = agent_settings.ml_api_url


def set_semaphore_factory(factory: "RedisSemaphoreFactory"):
    """Injeta a factory de semaphores Redis (chamado no lifespan)."""
    global _sem_factory
    _sem_factory = factory


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
    from projects.agent.resilience.circuit_breaker import get_registry, CircuitOpenError

    timeout = kwargs.pop("timeout", agent_settings.ml_api_timeout)
    headers = kwargs.pop("headers", {}) or {}
    headers = {"X-Agent-Version": agent_settings.agent_version, **headers}

    client = _ml_http_client

    if client is None:
        return tool_error(
            "UNAVAILABLE",
            "ML HTTP client nao inicializado. Verifique o lifespan da app.",
        )

    # Semaphore distribuído (Redis) ou fallback in-memory
    if _sem_factory is not None:
        redis_sem = _sem_factory.semaphore(
            f"ml_api:{account_id or 'global'}",
            max_concurrent=5,
            ttl=120,
        )
        acquired = await redis_sem.acquire()
        if not acquired:
            return tool_error("UNAVAILABLE", "Limite de chamadas ML API atingido.", retryable=True)
    else:
        redis_sem = None
        await _ml_api_semaphores[account_id or "global"].acquire()

    cb = get_registry().get("ml_api")

    # Verifica circuit breaker antes de tentar
    if cb.is_open:
        if redis_sem:
            await redis_sem.release()
        else:
            _ml_api_semaphores[account_id or "global"].release()
        return tool_error(
            "UNAVAILABLE",
            f"ML API temporariamente indisponivel (circuit breaker aberto). "
            f"Tente novamente em {cb.retry_after():.0f}s.",
            retryable=False,
        )

    start = time.monotonic()
    try:
        async def _do_request():
            resp = await getattr(client, method)(
                path,
                timeout=float(timeout),
                headers=headers,
                **kwargs,
            )
            resp.raise_for_status()
            return resp

        resp = await cb.call(_do_request)
        ml_api_latency.labels(
            method=method, path=path, status=str(resp.status_code),
        ).observe(time.monotonic() - start)
        return tool_success(resp.json())

    except CircuitOpenError as e:
        return tool_error("UNAVAILABLE", str(e), retryable=False)
    except httpx.TimeoutException:
        ml_api_latency.labels(method=method, path=path, status="timeout").observe(
            time.monotonic() - start
        )
        tool_errors_total.labels(tool_name="ml_api_call", error_code="TIMEOUT").inc()
        return tool_error(
            "TIMEOUT",
            f"ML API timeout em {path}. Tente novamente em instantes.",
            retryable=True,
        )
    except httpx.HTTPStatusError as e:
        ml_api_latency.labels(
            method=method, path=path, status=str(e.response.status_code),
        ).observe(time.monotonic() - start)
        tool_errors_total.labels(tool_name="ml_api_call", error_code="HTTP_ERROR").inc()
        return tool_error(
            "HTTP_ERROR",
            f"ML API retornou {e.response.status_code} em {path}.",
            retryable=e.response.status_code >= 500,
        )
    except httpx.ConnectError:
        ml_api_latency.labels(
            method=method, path=path, status="connect_error",
        ).observe(time.monotonic() - start)
        tool_errors_total.labels(tool_name="ml_api_call", error_code="UNAVAILABLE").inc()
        return tool_error(
            "UNAVAILABLE",
            "ML API indisponivel. Verifique se o servico esta rodando.",
            retryable=True,
        )
    finally:
        # Libera semaphore independente do resultado
        if redis_sem:
            await redis_sem.release()
        else:
            _ml_api_semaphores[account_id or "global"].release()
