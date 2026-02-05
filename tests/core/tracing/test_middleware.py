# tests/core/tracing/test_middleware.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from shared.core.tracing.middleware import TraceMiddleware
from shared.core.tracing.context import get_trace_context


def test_middleware_generates_trace_id_when_not_provided():
    """Middleware deve gerar trace_id quando não fornecido"""
    app = FastAPI()
    app.add_middleware(TraceMiddleware)

    @app.get("/test")
    async def test_endpoint():
        context = get_trace_context()
        return {"trace_id": context["trace_id"]}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200
    trace_id = response.json()["trace_id"]
    assert trace_id is not None
    assert len(trace_id) == 36  # UUID format


def test_middleware_uses_provided_trace_id():
    """Middleware deve usar X-Trace-ID do header quando fornecido"""
    app = FastAPI()
    app.add_middleware(TraceMiddleware)

    @app.get("/test")
    async def test_endpoint():
        context = get_trace_context()
        return {"trace_id": context["trace_id"]}

    client = TestClient(app)
    response = client.get("/test", headers={"X-Trace-ID": "custom-trace-123"})

    assert response.status_code == 200
    assert response.json()["trace_id"] == "custom-trace-123"


def test_middleware_adds_trace_id_to_response_header():
    """Middleware deve adicionar X-Trace-ID no header da resposta"""
    app = FastAPI()
    app.add_middleware(TraceMiddleware)

    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    client = TestClient(app)
    response = client.get("/test")

    assert "X-Trace-ID" in response.headers
    assert len(response.headers["X-Trace-ID"]) == 36


def test_middleware_logs_request_and_response():
    """Middleware deve logar request received e response sent"""
    app = FastAPI()
    app.add_middleware(TraceMiddleware)

    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    with patch('app.core.tracing.middleware.log_orchestrator_request_received') as mock_request:
        with patch('app.core.tracing.middleware.log_orchestrator_response_sent') as mock_response:
            client = TestClient(app)
            response = client.get("/test")

            assert response.status_code == 200
            mock_request.assert_called_once()
            mock_response.assert_called_once()


def test_middleware_logs_error_on_exception():
    """Middleware deve logar erro quando exceção ocorre"""
    app = FastAPI()
    app.add_middleware(TraceMiddleware)

    @app.get("/test")
    async def failing_endpoint():
        raise ValueError("Test error")

    with patch('app.core.tracing.middleware.log_orchestrator_request_failed') as mock_failed:
        client = TestClient(app)

        with pytest.raises(ValueError):
            client.get("/test")

        mock_failed.assert_called_once()
