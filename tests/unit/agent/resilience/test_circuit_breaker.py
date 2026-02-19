"""Testes unitários do CircuitBreaker."""
import asyncio
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock

from projects.agent.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
    get_registry,
    set_registry,
)


@pytest.fixture
def fast_config():
    """Config com timeout rápido para testes."""
    return CircuitBreakerConfig(
        failure_threshold=2,
        success_threshold=1,
        open_timeout=0.05,  # 50ms
    )


@pytest.fixture
def fast_cb(fast_config):
    return CircuitBreaker("test_dep", fast_config)


class TestCircuitBreakerStates:
    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self, fast_cb):
        assert fast_cb.state == CircuitState.CLOSED
        assert not fast_cb.is_open

    @pytest.mark.asyncio
    async def test_success_keeps_closed(self, fast_cb):
        result = await fast_cb.call(AsyncMock(return_value="ok"))
        assert result == "ok"
        assert fast_cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_single_failure_stays_closed(self, fast_cb):
        async def fail():
            raise ConnectionError("falha")

        with pytest.raises(ConnectionError):
            await fast_cb.call(fail)
        assert fast_cb.state == CircuitState.CLOSED  # threshold=2, apenas 1 falha

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self, fast_cb):
        async def fail():
            raise ConnectionError("falha")

        for _ in range(2):
            with pytest.raises(ConnectionError):
                await fast_cb.call(fail)

        assert fast_cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_raises_circuit_open_error(self, fast_cb):
        async def fail():
            raise ConnectionError("falha")

        # Abre o circuito
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await fast_cb.call(fail)

        # Fail-fast sem tentar a funcao real
        with pytest.raises(CircuitOpenError) as exc_info:
            await fast_cb.call(AsyncMock(return_value="ok"))

        assert exc_info.value.dependency == "test_dep"

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(self, fast_cb):
        async def fail():
            raise ConnectionError("falha")

        for _ in range(2):
            with pytest.raises(ConnectionError):
                await fast_cb.call(fail)

        assert fast_cb.state == CircuitState.OPEN

        await asyncio.sleep(0.1)  # Aguarda timeout (50ms)
        await fast_cb._check_state()
        assert fast_cb.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self, fast_cb):
        async def fail():
            raise ConnectionError("falha")

        for _ in range(2):
            with pytest.raises(ConnectionError):
                await fast_cb.call(fail)

        await asyncio.sleep(0.1)

        result = await fast_cb.call(AsyncMock(return_value="recuperado"))
        assert result == "recuperado"
        assert fast_cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self, fast_cb):
        async def fail():
            raise ConnectionError("falha")

        for _ in range(2):
            with pytest.raises(ConnectionError):
                await fast_cb.call(fail)

        await asyncio.sleep(0.1)

        with pytest.raises(ConnectionError):
            await fast_cb.call(fail)

        assert fast_cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self, fast_cb):
        async def fail():
            raise ConnectionError("falha")

        # Uma falha (threshold=2, nao abre)
        with pytest.raises(ConnectionError):
            await fast_cb.call(fail)

        # Sucesso reseta contador
        await fast_cb.call(AsyncMock(return_value="ok"))

        # Outra falha — ainda nao abre (contador zerado)
        with pytest.raises(ConnectionError):
            await fast_cb.call(fail)

        assert fast_cb.state == CircuitState.CLOSED

    def test_reset_forces_closed(self, fast_cb):
        fast_cb._state = CircuitState.OPEN
        fast_cb._failure_count = 10
        fast_cb.reset()
        assert fast_cb.state == CircuitState.CLOSED
        assert fast_cb._failure_count == 0


class TestCircuitBreakerRegistry:
    def test_get_creates_breaker(self):
        reg = CircuitBreakerRegistry()
        cb = reg.get("ml_api")
        assert isinstance(cb, CircuitBreaker)
        assert cb.name == "ml_api"

    def test_get_returns_same_instance(self):
        reg = CircuitBreakerRegistry()
        cb1 = reg.get("ml_api")
        cb2 = reg.get("ml_api")
        assert cb1 is cb2

    def test_status(self):
        reg = CircuitBreakerRegistry()
        reg.get("ml_api")
        reg.get("fb_api")
        status = reg.status()
        assert status["ml_api"] == "closed"
        assert status["fb_api"] == "closed"

    def test_reset_all(self):
        reg = CircuitBreakerRegistry()
        reg.get("ml_api")._state = CircuitState.OPEN
        reg.reset_all()
        assert reg.get("ml_api").state == CircuitState.CLOSED


class TestGlobalRegistry:
    def test_set_and_get_registry(self):
        original = get_registry()
        new_reg = CircuitBreakerRegistry()
        set_registry(new_reg)
        assert get_registry() is new_reg
        # Restaurar
        set_registry(original)
