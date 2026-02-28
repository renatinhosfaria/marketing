from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.fb_ads_router import fb_ads_api_router


def test_agent_query_route_exists() -> None:
    app = FastAPI()
    app.include_router(fb_ads_api_router, prefix="/api/v1")

    client = TestClient(app)
    response = client.post(
        "/api/v1/facebook-ads/agent/query",
        json={"prompt": "listar campanhas", "sql": "SELECT 1"},
    )

    assert response.status_code != 404
