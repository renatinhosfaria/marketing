"""
Testes de integracao do Memory Store (PostgresStore + busca semantica mockada).

Testa:
  - Namespace contract: save e recall usam mesmo namespace
  - StoreNamespace: formatacao de tuplas
  - Diferentes namespaces para diferentes users/accounts
  - Busca semantica retorna resultados relevantes
  - Store vazio retorna lista vazia
  - create_store_cm inicializa pool e chama setup
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from projects.agent.memory.namespaces import StoreNamespace


@pytest.mark.asyncio
async def test_store_namespace_user_profile():
    """StoreNamespace.user_profile retorna tupla correta."""
    ns = StoreNamespace.user_profile("u1")
    assert ns == ("u1", "profile")


@pytest.mark.asyncio
async def test_store_namespace_account_patterns():
    """StoreNamespace.account_patterns retorna tupla correta."""
    ns = StoreNamespace.account_patterns("u1", "a1")
    assert ns == ("u1", "a1", "patterns")


@pytest.mark.asyncio
async def test_store_namespace_account_actions():
    """StoreNamespace.account_actions retorna tupla correta."""
    ns = StoreNamespace.account_actions("u1", "a1")
    assert ns == ("u1", "a1", "action_history")


@pytest.mark.asyncio
async def test_store_namespace_account_insights():
    """StoreNamespace.account_insights retorna tupla correta."""
    ns = StoreNamespace.account_insights("u1", "a1")
    assert ns == ("u1", "a1", "insights")


@pytest.mark.asyncio
async def test_store_namespace_campaign_insights():
    """StoreNamespace.campaign_insights retorna tupla com campanha."""
    ns = StoreNamespace.campaign_insights("u1", "a1", "c1")
    assert ns == ("u1", "a1", "c1", "insights")


@pytest.mark.asyncio
async def test_store_namespace_isolation():
    """Namespaces de users/accounts diferentes sao distintos."""
    ns_user1 = StoreNamespace.account_patterns("u1", "a1")
    ns_user2 = StoreNamespace.account_patterns("u2", "a1")
    ns_account2 = StoreNamespace.account_patterns("u1", "a2")

    assert ns_user1 != ns_user2
    assert ns_user1 != ns_account2
    assert ns_user2 != ns_account2


@pytest.mark.asyncio
async def test_store_save_and_recall_same_namespace():
    """save_insight e recall_insights usam o mesmo namespace para mesma conta."""
    mock_store = AsyncMock()
    mock_store.aput = AsyncMock()
    mock_store.asearch = AsyncMock(return_value=[])

    config = {
        "configurable": {
            "user_id": "u1",
            "account_id": "a1",
            "thread_id": "t1",
        }
    }

    # Save — chamar a funcao interna diretamente passando store
    from projects.agent.tools.shared import save_insight
    await save_insight.coroutine(
        insight_text="Teste de namespace",
        context="Integracao",
        category="performance_pattern",
        store=mock_store,
        config=config,
    )

    save_ns = mock_store.aput.call_args[0][0]

    # Recall
    from projects.agent.tools.shared import recall_insights
    await recall_insights.coroutine(
        query="teste",
        store=mock_store,
        config=config,
    )

    recall_ns = mock_store.asearch.call_args[0][0]

    # Ambos devem usar o mesmo namespace
    assert save_ns == recall_ns
    assert save_ns == StoreNamespace.account_patterns("u1", "a1")


@pytest.mark.asyncio
async def test_store_semantic_search_returns_relevant():
    """Busca semantica retorna itens relevantes quando existem."""
    mock_item = MagicMock()
    mock_item.value = {
        "insight_text": "CPL sobe nas segundas-feiras",
        "context": "Analise semanal",
        "category": "performance_pattern",
        "discovered_at": "2026-02-10T10:00:00",
    }

    mock_store = AsyncMock()
    mock_store.asearch = AsyncMock(return_value=[mock_item])

    config = {
        "configurable": {
            "user_id": "u1",
            "account_id": "a1",
            "thread_id": "t1",
        }
    }

    from projects.agent.tools.shared import recall_insights
    result = await recall_insights.coroutine(
        query="tendencia de CPL por dia da semana",
        store=mock_store,
        config=config,
    )

    assert result["ok"] is True
    assert len(result["data"]) == 1
    assert "CPL" in result["data"][0]["insight_text"]


@pytest.mark.asyncio
async def test_create_store_cm_calls_setup():
    """create_store_cm inicializa pool, cria Store e chama setup()."""
    mock_pool_instance = AsyncMock()
    mock_pool_instance.open = AsyncMock()
    mock_pool_instance.close = AsyncMock()

    mock_store_instance = AsyncMock()
    mock_store_instance.setup = AsyncMock()

    with patch("projects.agent.memory.store.AsyncConnectionPool", return_value=mock_pool_instance) as mock_pool_cls, \
         patch("projects.agent.memory.store.AsyncPostgresStore", return_value=mock_store_instance) as mock_store_cls, \
         patch("projects.agent.memory.store.settings") as mock_settings, \
         patch("projects.agent.memory.store.agent_settings") as mock_agent_settings:
        mock_settings.database_url = "postgresql://test"
        mock_agent_settings.openai_api_key = None
        mock_agent_settings.store_embedding_model = "openai:text-embedding-3-small"
        mock_agent_settings.store_embedding_dims = 1536

        from projects.agent.memory.store import create_store_cm
        async with create_store_cm() as store:
            pass

    mock_pool_instance.open.assert_awaited_once()
    mock_store_instance.setup.assert_awaited_once()
    mock_pool_instance.close.assert_awaited_once()
    assert store == mock_store_instance
