from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.fb_ads_router import fb_ads_api_router
from projects.facebook_ads.services.agent_query_service import AgentQueryService
from shared.db.session import get_db


def test_agent_query_route_exists() -> None:
    app = FastAPI()
    app.include_router(fb_ads_api_router, prefix="/api/v1")

    client = TestClient(app)
    response = client.post(
        "/api/v1/facebook-ads/agent/query",
        json={"prompt": "listar campanhas", "sql": "SELECT 1"},
    )

    assert response.status_code != 404


def test_agent_query_accepts_prompt_without_sql(monkeypatch) -> None:
    async def _fake_db():
        yield object()

    async def _fake_execute(self, prompt: str, sql: str | None, requested_by: str = "fbads"):
        return {
            "operationType": "SELECT",
            "sqlExecuted": "SELECT 1",
            "rowsAffected": 0,
            "rows": [],
            "durationMs": 1,
        }

    monkeypatch.setattr(AgentQueryService, "execute_sql", _fake_execute)

    app = FastAPI()
    app.include_router(fb_ads_api_router, prefix="/api/v1")
    app.dependency_overrides[get_db] = _fake_db

    client = TestClient(app)
    response = client.post(
        "/api/v1/facebook-ads/agent/query",
        json={"prompt": "listar top 5 campanhas por spend da config 1"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["sqlExecuted"] == "SELECT 1"
