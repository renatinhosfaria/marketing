"""
Testes das tools compartilhadas (save_insight, recall_insights).

Testa:
  - save_insight: salva no Store com namespace correto
  - recall_insights: busca semantica no Store
  - Namespace correto baseado em user_id + account_id
  - ToolResult contract (ok/data/error)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from projects.agent.memory.namespaces import StoreNamespace


@pytest.mark.asyncio
async def test_save_insight_success():
    """save_insight salva insight no Store com namespace correto."""
    mock_store = AsyncMock()
    mock_store.aput = AsyncMock()

    config = {
        "configurable": {
            "user_id": "u1",
            "account_id": "a1",
            "thread_id": "t1",
        }
    }

    from projects.agent.tools.shared import save_insight
    result = await save_insight.coroutine(
        insight_text="CPL tende a subir nas segundas-feiras",
        context="Analise semanal de performance",
        category="performance_pattern",
        store=mock_store,
        config=config,
    )

    assert result["ok"] is True
    assert "salvo" in result["data"]["message"].lower()
    mock_store.aput.assert_awaited_once()

    # Verificar namespace
    call_args = mock_store.aput.call_args
    namespace = call_args[0][0]
    expected_ns = StoreNamespace.account_patterns("u1", "a1")
    assert namespace == expected_ns


@pytest.mark.asyncio
async def test_save_insight_categories():
    """save_insight aceita todas as categorias validas."""
    categories = ["performance_pattern", "audience_preference", "creative_learning"]

    for cat in categories:
        mock_store = AsyncMock()
        mock_store.aput = AsyncMock()

        config = {
            "configurable": {
                "user_id": "u1",
                "account_id": "a1",
                "thread_id": "t1",
            }
        }

        from projects.agent.tools.shared import save_insight
        result = await save_insight.coroutine(
            insight_text=f"Insight de {cat}",
            context="Teste",
            category=cat,
            store=mock_store,
            config=config,
        )

        assert result["ok"] is True


@pytest.mark.asyncio
async def test_recall_insights_success():
    """recall_insights busca semantica no Store e retorna resultados."""
    mock_item_1 = MagicMock()
    mock_item_1.value = {
        "insight_text": "CPL sobe nas segundas",
        "context": "Analise semanal",
        "category": "performance_pattern",
    }
    mock_item_2 = MagicMock()
    mock_item_2.value = {
        "insight_text": "Publico jovem responde melhor a video",
        "context": "Teste A/B",
        "category": "creative_learning",
    }

    mock_store = AsyncMock()
    mock_store.asearch = AsyncMock(return_value=[mock_item_1, mock_item_2])

    config = {
        "configurable": {
            "user_id": "u1",
            "account_id": "a1",
            "thread_id": "t1",
        }
    }

    from projects.agent.tools.shared import recall_insights
    result = await recall_insights.coroutine(
        query="tendencias de CPL",
        store=mock_store,
        config=config,
    )

    assert result["ok"] is True
    assert len(result["data"]) == 2
    assert result["data"][0]["insight_text"] == "CPL sobe nas segundas"


@pytest.mark.asyncio
async def test_recall_insights_empty():
    """recall_insights retorna lista vazia quando nao ha insights."""
    mock_store = AsyncMock()
    mock_store.asearch = AsyncMock(return_value=[])

    config = {
        "configurable": {
            "user_id": "u1",
            "account_id": "a1",
            "thread_id": "t1",
        }
    }

    from projects.agent.tools.shared import recall_insights
    result = await recall_insights.coroutine(
        query="algo inexistente",
        store=mock_store,
        config=config,
    )

    assert result["ok"] is True
    assert result["data"] == []


@pytest.mark.asyncio
async def test_recall_insights_uses_correct_namespace():
    """recall_insights busca no namespace correto do usuario/conta."""
    mock_store = AsyncMock()
    mock_store.asearch = AsyncMock(return_value=[])

    config = {
        "configurable": {
            "user_id": "u42",
            "account_id": "act_999",
            "thread_id": "t1",
        }
    }

    from projects.agent.tools.shared import recall_insights
    await recall_insights.coroutine(
        query="teste",
        store=mock_store,
        config=config,
    )

    call_args = mock_store.asearch.call_args
    namespace = call_args[0][0]
    expected_ns = StoreNamespace.account_patterns("u42", "act_999")
    assert namespace == expected_ns
