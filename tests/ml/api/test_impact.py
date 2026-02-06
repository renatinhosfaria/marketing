"""Tests for Impact Analysis API endpoint."""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from projects.ml.api.impact import router, ImpactAnalysisRequest


@pytest.fixture
def app():
    """Create test FastAPI app."""
    app = FastAPI()
    app.include_router(router, prefix="/impact")
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_impact_result():
    """Create mock ImpactResult."""
    from projects.ml.algorithms.models.causal.impact_analyzer import ImpactResult

    return ImpactResult(
        entity_type='campaign',
        entity_id='123456',
        change_date=datetime(2025, 1, 8),
        change_type='budget_change',
        metric_changes={'cpl': -15.5, 'leads': 28.3},
        significance={'cpl': 0.95, 'leads': 0.92},
        is_significant={'cpl': True, 'leads': True},
        effect_sizes={'cpl': -0.85, 'leads': 0.72},
        overall_impact='positive',
        recommendation='Budget change was beneficial. Consider further scaling if metrics remain stable.',
        window_before=7,
        window_after=7,
        analyzed_at=datetime(2025, 1, 15, 10, 30, 0),
    )


class TestImpactAnalysisRequest:
    """Tests for request validation."""

    def test_valid_request(self):
        """Should accept valid request."""
        request = ImpactAnalysisRequest(
            config_id=1,
            entity_type='campaign',
            entity_id='123456',
            change_date=datetime(2025, 1, 8),
            change_type='budget_change',
        )
        assert request.config_id == 1
        assert request.entity_type == 'campaign'
        assert request.window_before == 7
        assert request.window_after == 7

    def test_invalid_entity_type(self):
        """Should reject invalid entity_type."""
        with pytest.raises(ValueError):
            ImpactAnalysisRequest(
                config_id=1,
                entity_type='invalid',
                entity_id='123456',
                change_date=datetime(2025, 1, 8),
                change_type='budget_change',
            )

    def test_invalid_change_type(self):
        """Should reject invalid change_type."""
        with pytest.raises(ValueError):
            ImpactAnalysisRequest(
                config_id=1,
                entity_type='campaign',
                entity_id='123456',
                change_date=datetime(2025, 1, 8),
                change_type='invalid_change',
            )

    def test_window_constraints(self):
        """Should enforce window constraints."""
        # Valid range
        request = ImpactAnalysisRequest(
            config_id=1,
            entity_type='campaign',
            entity_id='123456',
            change_date=datetime(2025, 1, 8),
            change_type='budget_change',
            window_before=3,
            window_after=30,
        )
        assert request.window_before == 3
        assert request.window_after == 30

        # Below minimum
        with pytest.raises(ValueError):
            ImpactAnalysisRequest(
                config_id=1,
                entity_type='campaign',
                entity_id='123456',
                change_date=datetime(2025, 1, 8),
                change_type='budget_change',
                window_before=2,  # min is 3
            )

        # Above maximum
        with pytest.raises(ValueError):
            ImpactAnalysisRequest(
                config_id=1,
                entity_type='campaign',
                entity_id='123456',
                change_date=datetime(2025, 1, 8),
                change_type='budget_change',
                window_after=31,  # max is 30
            )


class TestImpactAnalyzeEndpoint:
    """Tests for POST /impact/analyze endpoint."""

    @pytest.mark.asyncio
    async def test_analyze_success(self, app, mock_impact_result):
        """Should return impact analysis result."""
        with patch('projects.ml.api.impact.get_db') as mock_get_db, \
             patch('projects.ml.api.impact.get_impact_analyzer') as mock_get_analyzer:

            mock_session = AsyncMock()
            mock_get_db.return_value = mock_session

            mock_analyzer = AsyncMock()
            mock_analyzer.analyze_impact.return_value = mock_impact_result
            mock_get_analyzer.return_value = mock_analyzer

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test"
            ) as ac:
                response = await ac.post(
                    "/impact/analyze",
                    json={
                        "config_id": 1,
                        "entity_type": "campaign",
                        "entity_id": "123456",
                        "change_date": "2025-01-08T00:00:00",
                        "change_type": "budget_change",
                    },
                )

            assert response.status_code == 200
            data = response.json()
            assert data['entity_type'] == 'campaign'
            assert data['entity_id'] == '123456'
            assert data['overall_impact'] == 'positive'
            assert data['change_type'] == 'budget_change'
            assert len(data['metric_changes']) > 0
            assert 'recommendation' in data

    @pytest.mark.asyncio
    async def test_analyze_value_error_returns_400(self, app):
        """Should return 400 on ValueError."""
        with patch('projects.ml.api.impact.get_db') as mock_get_db, \
             patch('projects.ml.api.impact.get_impact_analyzer') as mock_get_analyzer:

            mock_session = AsyncMock()
            mock_get_db.return_value = mock_session

            mock_analyzer = AsyncMock()
            mock_analyzer.analyze_impact.side_effect = ValueError("No data for entity")
            mock_get_analyzer.return_value = mock_analyzer

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test"
            ) as ac:
                response = await ac.post(
                    "/impact/analyze",
                    json={
                        "config_id": 1,
                        "entity_type": "campaign",
                        "entity_id": "nonexistent",
                        "change_date": "2025-01-08T00:00:00",
                        "change_type": "budget_change",
                    },
                )

            assert response.status_code == 400
            assert "No data for entity" in response.json()['detail']

    def test_analyze_invalid_request(self, client):
        """Should return 422 on invalid request."""
        response = client.post(
            "/impact/analyze",
            json={
                "config_id": 1,
                "entity_type": "invalid_type",
                "entity_id": "123456",
                "change_date": "2025-01-08T00:00:00",
                "change_type": "budget_change",
            },
        )
        assert response.status_code == 422
