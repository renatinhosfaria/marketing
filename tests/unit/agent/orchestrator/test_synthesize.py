"""Testes para synthesize node."""
import pytest
from unittest.mock import AsyncMock, patch


class TestSynthesize:
    """Testes para o no synthesize."""

    def test_synthesize_import(self):
        """synthesize deve ser importavel."""
        from app.agent.orchestrator.nodes.synthesize import synthesize
        assert synthesize is not None

    def test_format_results_for_synthesis(self):
        """format_results_for_synthesis deve formatar resultados."""
        from app.agent.orchestrator.nodes.synthesize import format_results_for_synthesis

        results = {
            "classification": {
                "success": True,
                "data": {"response": "Analise de tiers"},
                "tool_calls": ["get_classifications"]
            },
            "anomaly": {
                "success": True,
                "data": {"response": "Problemas encontrados"},
                "tool_calls": ["get_anomalies"]
            }
        }

        formatted = format_results_for_synthesis(results)

        assert "classification" in formatted.lower() or "classificacao" in formatted.lower()
        assert "anomaly" in formatted.lower() or "anomalia" in formatted.lower()

    def test_prioritize_results(self):
        """prioritize_results deve ordenar por prioridade."""
        from app.agent.orchestrator.nodes.synthesize import prioritize_results

        results = {
            "classification": {"success": True},
            "anomaly": {"success": True},
            "recommendation": {"success": True}
        }

        ordered = prioritize_results(results)

        # Anomaly deve vir primeiro (prioridade 1)
        assert ordered[0][0] == "anomaly"

    def test_get_synthesis_prompt(self):
        """get_synthesis_prompt deve retornar prompt valido."""
        from app.agent.orchestrator.prompts import get_synthesis_prompt

        prompt = get_synthesis_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "sintese" in prompt.lower() or "resumo" in prompt.lower()
