"""
Circuit breaker para dependências externas do Agent.

Implementação de circuit breaker com estados CLOSED → OPEN → HALF_OPEN → CLOSED.
Protege ML API, FB API e Store de cascatas de falha.

Estados:
  CLOSED    — operação normal; erros são contados
  OPEN      — fail-fast imediato; aguarda timeout para testar recuperação
  HALF_OPEN — permite uma chamada de teste; sucesso fecha, falha reabre
"""

import asyncio
import time
from enum import Enum
from dataclasses import dataclass
from typing import Callable, TypeVar, Awaitable
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuração do circuit breaker."""
    failure_threshold: int = 5       # Falhas consecutivas para abrir
    success_threshold: int = 2       # Sucessos consecutivos para fechar de HALF_OPEN
    open_timeout: float = 60.0       # Segundos em OPEN antes de testar (HALF_OPEN)
    half_open_max_calls: int = 1     # Chamadas permitidas em HALF_OPEN


class CircuitOpenError(Exception):
    """Levantado quando o circuito está aberto (fail-fast)."""

    def __init__(self, dependency: str, retry_after: float):
        self.dependency = dependency
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker OPEN para '{dependency}'. "
            f"Tente novamente em {retry_after:.0f}s."
        )


class CircuitBreaker:
    """
    Circuit breaker para uma dependência externa.

    Thread-safe para uso em async (asyncio.Lock interno).
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    def retry_after(self) -> float:
        """Segundos restantes até poder tentar de novo."""
        if self._last_failure_time is None:
            return 0.0
        elapsed = time.monotonic() - self._last_failure_time
        return max(0.0, self._config.open_timeout - elapsed)

    async def _check_state(self):
        """Transiciona OPEN → HALF_OPEN se o timeout expirou."""
        if self._state == CircuitState.OPEN and self.retry_after() <= 0:
            async with self._lock:
                if self._state == CircuitState.OPEN:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._success_count = 0
                    logger.info(
                        "circuit_breaker.half_open",
                        dependency=self.name,
                    )

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """
        Executa func protegida pelo circuit breaker.

        Levanta CircuitOpenError se o circuito estiver aberto.
        """
        await self._check_state()

        if self._state == CircuitState.OPEN:
            raise CircuitOpenError(self.name, self.retry_after())

        if self._state == CircuitState.HALF_OPEN:
            async with self._lock:
                if self._half_open_calls >= self._config.half_open_max_calls:
                    raise CircuitOpenError(self.name, self.retry_after())
                self._half_open_calls += 1

        try:
            result = await func()
            await self._on_success()
            return result
        except CircuitOpenError:
            raise
        except Exception as exc:
            await self._on_failure(exc)
            raise

    async def _on_success(self):
        """Registra sucesso; fecha circuito se em HALF_OPEN."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info(
                        "circuit_breaker.closed",
                        dependency=self.name,
                        reason="recovery_success",
                    )
                    _set_metric(self.name, 0)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0  # reset em sucesso

    async def _on_failure(self, exc: Exception):
        """Registra falha; abre circuito se threshold atingido."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
                logger.warning(
                    "circuit_breaker.reopened",
                    dependency=self.name,
                    error=str(exc),
                )
                _set_metric(self.name, 1)

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.error(
                        "circuit_breaker.opened",
                        dependency=self.name,
                        failure_count=self._failure_count,
                        error=str(exc),
                    )
                    _set_metric(self.name, 1)

    def reset(self):
        """Força reset para CLOSED (para testes ou operação manual)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0


def _set_metric(dependency: str, value: float):
    """Atualiza métrica Prometheus — falha silenciosa se não disponível."""
    try:
        from projects.agent.observability.metrics import dependency_circuit_open
        dependency_circuit_open.labels(dependency=dependency).set(value)
    except Exception:
        pass


class _NoOpCircuitBreaker:
    """Circuit breaker no-op — sempre CLOSED, passa tudo (usado quando disabled)."""

    name = "noop"
    state = CircuitState.CLOSED
    is_open = False

    def retry_after(self) -> float:
        return 0.0

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        return await func()

    def reset(self):
        pass


_NOOP_CB = _NoOpCircuitBreaker()


class CircuitBreakerRegistry:
    """
    Registro de circuit breakers por dependência.

    Uma instância por aplicação, inicializada no lifespan.

    enabled=False: todas as chamadas usam _NoOpCircuitBreaker (rollback rapido).
    """

    def __init__(
        self,
        default_config: CircuitBreakerConfig | None = None,
        enabled: bool = True,
    ):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._default_config = default_config or CircuitBreakerConfig()
        self.enabled = enabled

    def get(self, dependency: str) -> "CircuitBreaker | _NoOpCircuitBreaker":
        """Retorna ou cria o circuit breaker para a dependência.

        Retorna no-op se registry desabilitado (feature flag).
        """
        if not self.enabled:
            return _NOOP_CB
        if dependency not in self._breakers:
            self._breakers[dependency] = CircuitBreaker(
                dependency, self._default_config
            )
        return self._breakers[dependency]

    def status(self) -> dict[str, str]:
        """Retorna estado de todos os circuit breakers."""
        if not self.enabled:
            return {"status": "disabled"}
        return {name: cb.state.value for name, cb in self._breakers.items()}

    def reset_all(self):
        """Reseta todos (para testes)."""
        for cb in self._breakers.values():
            cb.reset()


# Registry global — inicializado no lifespan via set_registry()
_registry: CircuitBreakerRegistry | None = None


def get_registry() -> CircuitBreakerRegistry:
    """Retorna o registry global; cria instância default se não inicializado."""
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry


def set_registry(registry: CircuitBreakerRegistry):
    """Configura o registry global (chamado no lifespan da app)."""
    global _registry
    _registry = registry
