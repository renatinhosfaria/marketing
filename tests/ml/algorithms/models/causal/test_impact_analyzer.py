"""Tests for Impact Analyzer."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock


class TestImpactAnalyzer:
    """Tests for ImpactAnalyzer class."""

    @pytest.fixture
    def before_after_data(self):
        """Generate before/after data with clear difference."""
        dates = pd.date_range(start='2025-01-01', periods=14, freq='D')

        # Before: higher CPL
        before_cpl = [60, 58, 62, 55, 57, 61, 59]
        # After: lower CPL (improvement)
        after_cpl = [45, 42, 48, 44, 46, 43, 47]

        return pd.DataFrame({
            'date': dates,
            'cpl': before_cpl + after_cpl,
            'leads': [5, 6, 4, 5, 5, 4, 6, 8, 9, 7, 8, 8, 9, 7],
            'ctr': [1.5] * 14,
            'spend': [300] * 14,
        })

    @pytest.mark.asyncio
    async def test_analyze_impact_detects_improvement(self, before_after_data):
        """Should detect positive impact when CPL decreases."""
        from projects.ml.algorithms.models.causal.impact_analyzer import ImpactAnalyzer

        mock_data_service = AsyncMock()
        mock_data_service.get_entity_daily_data.return_value = before_after_data

        analyzer = ImpactAnalyzer()
        result = await analyzer.analyze_impact(
            entity_type='campaign',
            entity_id='123',
            change_date=datetime(2025, 1, 8),
            change_type='budget_change',
            data_service=mock_data_service,
            config_id=1,
            window_before=7,
            window_after=7,
        )

        assert result.overall_impact == 'positive'
        assert result.metric_changes['cpl'] < 0  # CPL decreased
        assert result.is_significant['cpl'] == True

    @pytest.mark.asyncio
    async def test_analyze_impact_returns_inconclusive_with_small_change(self):
        """Should return inconclusive when changes are not significant."""
        from projects.ml.algorithms.models.causal.impact_analyzer import ImpactAnalyzer

        # Data with minimal difference
        dates = pd.date_range(start='2025-01-01', periods=14, freq='D')
        np.random.seed(42)
        data = pd.DataFrame({
            'date': dates,
            'cpl': [50 + np.random.randn() for _ in range(14)],
            'leads': [5] * 14,
            'ctr': [1.5] * 14,
            'spend': [300] * 14,
        })

        mock_data_service = AsyncMock()
        mock_data_service.get_entity_daily_data.return_value = data

        analyzer = ImpactAnalyzer()
        result = await analyzer.analyze_impact(
            entity_type='campaign',
            entity_id='123',
            change_date=datetime(2025, 1, 8),
            change_type='budget_change',
            data_service=mock_data_service,
            config_id=1,
        )

        assert result.overall_impact in ['neutral', 'inconclusive']
