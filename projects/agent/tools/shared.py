"""
Tools compartilhadas disponiveis a todos os agentes.

- save_insight: salva insight na memoria de longo prazo (Store)
- recall_insights: busca semantica de insights relevantes
"""

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from typing import Annotated, Literal
from uuid import uuid4
from datetime import datetime, timezone

from projects.agent.memory.namespaces import StoreNamespace
from projects.agent.tools.result import ToolResult, tool_success


@tool
async def save_insight(
    insight_text: str,
    context: str,
    category: Literal[
        "performance_pattern",
        "audience_preference",
        "creative_learning",
    ],
    store: Annotated[BaseStore, InjectedStore()],
    config: RunnableConfig,
) -> ToolResult:
    """Salva um insight descoberto para memoria de longo prazo.

    Args:
        insight_text: Texto descritivo do insight descoberto.
        context: Contexto adicional sobre como o insight foi descoberto.
        category: Categoria do insight (performance_pattern, audience_preference,
                  creative_learning).
    """
    cfg = config.get("configurable", {})
    user_id = cfg["user_id"]
    account_id = cfg["account_id"]

    namespace = StoreNamespace.account_patterns(user_id, account_id)

    await store.aput(
        namespace,
        key=str(uuid4()),
        value={
            "insight_text": insight_text,
            "context": context,
            "category": category,
            "discovered_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    return tool_success({"message": f"Insight salvo: {insight_text[:80]}..."})


@tool
async def recall_insights(
    query: str,
    store: Annotated[BaseStore, InjectedStore()],
    config: RunnableConfig,
) -> ToolResult:
    """Busca insights relevantes na memoria de longo prazo (busca semantica).

    Args:
        query: Texto de busca para encontrar insights relacionados.
    """
    cfg = config.get("configurable", {})
    user_id = cfg["user_id"]
    account_id = cfg["account_id"]

    # Busca no MESMO namespace onde save_insight grava
    namespace = StoreNamespace.account_patterns(user_id, account_id)

    results = await store.asearch(
        namespace,
        query=query,
        limit=5,
    )

    return tool_success([item.value for item in results])
