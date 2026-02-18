"""
Testes do no Supervisor — classificacao de intencao + dispatch via Send().

Testa:
  - Roteamento correto para agentes baseado em RoutingDecision
  - Fallback quando remaining_steps <= 3
  - Resposta default quando nenhum agente e selecionado
  - Propagacao do scope para Send()
  - Estrutura correta dos Send() retornados
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Send, Command

from projects.agent.graph.routing import RoutingDecision, AnalysisScope


@pytest.mark.asyncio
async def test_supervisor_routes_to_health_on_anomaly():
    """Quando CPL sobe, supervisor deve rotear para health_monitor e performance_analyst."""
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = RoutingDecision(
        reasoning="CPL alto indica anomalia",
        selected_agents=["health_monitor", "performance_analyst"],
        urgency="high",
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    state = {
        "messages": [HumanMessage(content="Meu CPL subiu 40%")],
        "user_context": {"user_id": "u1", "account_id": "a1", "account_name": "", "timezone": "America/Sao_Paulo"},
        "agent_reports": [],
        "routing_decision": None,
        "pending_actions": [],
        "synthesis": None,
        "remaining_steps": 100,
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.graph.supervisor.get_model", return_value=mock_model):
        from projects.agent.graph.supervisor import supervisor_node
        result = await supervisor_node(state, config)

    assert isinstance(result, Command)
    sends = result.goto
    assert len(sends) == 2
    assert all(isinstance(s, Send) for s in sends)
    node_names = [s.node for s in sends]
    assert "health_monitor" in node_names
    assert "performance_analyst" in node_names


@pytest.mark.asyncio
async def test_supervisor_remaining_steps_limit():
    """Quando remaining_steps <= 3, retorna report de limite atingido."""
    state = {
        "messages": [HumanMessage(content="Qualquer pergunta")],
        "user_context": {"user_id": "u1", "account_id": "a1", "account_name": "", "timezone": "America/Sao_Paulo"},
        "agent_reports": [],
        "routing_decision": None,
        "pending_actions": [],
        "synthesis": None,
        "remaining_steps": 2,
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    from projects.agent.graph.supervisor import supervisor_node
    result = await supervisor_node(state, config)

    assert isinstance(result, dict)
    assert "agent_reports" in result
    assert result["agent_reports"][0]["agent_id"] == "supervisor"
    assert result["agent_reports"][0]["status"] == "completed"
    assert "messages" in result
    assert isinstance(result["messages"][0], AIMessage)


@pytest.mark.asyncio
async def test_supervisor_no_agents_selected():
    """Quando LLM nao seleciona agentes, retorna mensagem de ajuda."""
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = RoutingDecision(
        reasoning="Pergunta generica sem contexto de ads",
        selected_agents=[],
        urgency="low",
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    state = {
        "messages": [HumanMessage(content="Oi")],
        "user_context": {"user_id": "u1", "account_id": "a1", "account_name": "", "timezone": "America/Sao_Paulo"},
        "agent_reports": [],
        "routing_decision": None,
        "pending_actions": [],
        "synthesis": None,
        "remaining_steps": 100,
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.graph.supervisor.get_model", return_value=mock_model):
        from projects.agent.graph.supervisor import supervisor_node
        result = await supervisor_node(state, config)

    assert isinstance(result, dict)
    assert "messages" in result
    assert "ajudar" in result["messages"][0].content.lower()


@pytest.mark.asyncio
async def test_supervisor_propagates_scope_in_send():
    """Send() deve propagar scope como dict para os subgraphs."""
    scope = AnalysisScope(
        entity_type="campaign",
        entity_ids=["c1", "c2"],
        lookback_days=14,
        top_n=5,
    )
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = RoutingDecision(
        reasoning="Analise de performance solicitada",
        selected_agents=["performance_analyst"],
        urgency="medium",
        scope=scope,
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    state = {
        "messages": [HumanMessage(content="Analise minhas campanhas c1 e c2")],
        "user_context": {"user_id": "u1", "account_id": "a1", "account_name": "", "timezone": "America/Sao_Paulo"},
        "agent_reports": [],
        "routing_decision": None,
        "pending_actions": [],
        "synthesis": None,
        "remaining_steps": 100,
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.graph.supervisor.get_model", return_value=mock_model):
        from projects.agent.graph.supervisor import supervisor_node
        result = await supervisor_node(state, config)

    assert isinstance(result, Command)
    sends = result.goto
    assert len(sends) == 1
    send_arg = sends[0].arg
    assert "scope" in send_arg
    assert send_arg["scope"]["entity_ids"] == ["c1", "c2"]
    assert send_arg["scope"]["lookback_days"] == 14


@pytest.mark.asyncio
async def test_supervisor_multiple_agents_fanout():
    """Supervisor deve poder despachar multiplos agentes (fan-out)."""
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = RoutingDecision(
        reasoning="Analise completa solicitada",
        selected_agents=[
            "health_monitor",
            "performance_analyst",
            "creative_specialist",
            "forecast_scientist",
        ],
        urgency="high",
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    state = {
        "messages": [HumanMessage(content="Faca uma analise completa da minha conta")],
        "user_context": {"user_id": "u1", "account_id": "a1", "account_name": "", "timezone": "America/Sao_Paulo"},
        "agent_reports": [],
        "routing_decision": None,
        "pending_actions": [],
        "synthesis": None,
        "remaining_steps": 100,
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.graph.supervisor.get_model", return_value=mock_model):
        from projects.agent.graph.supervisor import supervisor_node
        result = await supervisor_node(state, config)

    assert isinstance(result, Command)
    sends = result.goto
    assert len(sends) == 4
    node_names = {s.node for s in sends}
    assert node_names == {
        "health_monitor",
        "performance_analyst",
        "creative_specialist",
        "forecast_scientist",
    }
