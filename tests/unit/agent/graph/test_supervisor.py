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

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.types import Send, Command

from projects.agent.graph.routing import RoutingDecision, AnalysisScope
from projects.agent.graph.supervisor import _build_context_messages, _build_reports_context


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


# --- Testes de contexto multi-turn ---


@pytest.mark.asyncio
async def test_supervisor_multi_turn_passes_history():
    """Com 3 mensagens no state, ainvoke recebe System + 3 mensagens (nao apenas a ultima)."""
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = RoutingDecision(
        reasoning="Previsoes das campanhas mencionadas",
        selected_agents=["forecast_scientist"],
        urgency="medium",
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    messages = [
        HumanMessage(content="Analise minhas campanhas c1 e c2"),
        AIMessage(content="Suas campanhas c1 e c2 estao com CPL estavel."),
        HumanMessage(content="E as previsoes?"),
    ]
    state = {
        "messages": messages,
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
        await supervisor_node(state, config)

    call_args = mock_structured.ainvoke.call_args[0][0]
    # System + 3 mensagens do historico
    assert len(call_args) == 4
    assert isinstance(call_args[0], SystemMessage)
    assert isinstance(call_args[1], HumanMessage)
    assert isinstance(call_args[2], AIMessage)
    assert isinstance(call_args[3], HumanMessage)
    assert call_args[3].content == "E as previsoes?"


@pytest.mark.asyncio
async def test_supervisor_long_conversation_applies_window():
    """Com 30 mensagens e window=5, ainvoke recebe System + 5 (nao todas)."""
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = RoutingDecision(
        reasoning="Pergunta generica",
        selected_agents=["health_monitor"],
        urgency="low",
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    # 30 mensagens alternando Human/AI
    messages = []
    for i in range(15):
        messages.append(HumanMessage(content=f"Pergunta {i}"))
        messages.append(AIMessage(content=f"Resposta {i}"))

    state = {
        "messages": messages,
        "user_context": {"user_id": "u1", "account_id": "a1", "account_name": "", "timezone": "America/Sao_Paulo"},
        "agent_reports": [],
        "routing_decision": None,
        "pending_actions": [],
        "synthesis": None,
        "remaining_steps": 100,
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.graph.supervisor.get_model", return_value=mock_model):
        with patch("projects.agent.graph.supervisor.MAX_HISTORY_MESSAGES", 5):
            from projects.agent.graph.supervisor import supervisor_node
            await supervisor_node(state, config)

    call_args = mock_structured.ainvoke.call_args[0][0]
    # System + 5 mensagens (window cortou as 25 mais antigas)
    assert len(call_args) == 6
    assert isinstance(call_args[0], SystemMessage)
    # A ultima mensagem do window deve ser a ultima do historico
    assert call_args[-1].content == "Resposta 14"
    # Ultimas 5 das 30: indices 25-29 = Resposta 12, Pergunta 13, Resposta 13, Pergunta 14, Resposta 14
    assert call_args[1].content == "Resposta 12"


@pytest.mark.asyncio
async def test_supervisor_includes_reports_context():
    """Com agent_reports no state, ainvoke recebe SystemMessage extra com resumo."""
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = RoutingDecision(
        reasoning="Previsoes solicitadas",
        selected_agents=["forecast_scientist"],
        urgency="medium",
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    state = {
        "messages": [HumanMessage(content="E as previsoes?")],
        "user_context": {"user_id": "u1", "account_id": "a1", "account_name": "", "timezone": "America/Sao_Paulo"},
        "agent_reports": [
            {
                "agent_id": "health_monitor",
                "status": "completed",
                "summary": "Detectadas 3 anomalias em campanhas ativas",
                "data": None,
                "confidence": 0.9,
            }
        ],
        "routing_decision": None,
        "pending_actions": [],
        "synthesis": None,
        "remaining_steps": 100,
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.graph.supervisor.get_model", return_value=mock_model):
        from projects.agent.graph.supervisor import supervisor_node
        await supervisor_node(state, config)

    call_args = mock_structured.ainvoke.call_args[0][0]
    # System prompt + reports context + 1 HumanMessage
    assert len(call_args) == 3
    assert isinstance(call_args[0], SystemMessage)  # SYSTEM_PROMPT
    assert isinstance(call_args[1], SystemMessage)  # reports context
    assert "health_monitor" in call_args[1].content
    assert "Detectadas 3 anomalias" in call_args[1].content
    assert isinstance(call_args[2], HumanMessage)


@pytest.mark.asyncio
async def test_supervisor_no_reports_no_extra_context():
    """Sem agent_reports, nenhum SystemMessage extra e adicionado."""
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = RoutingDecision(
        reasoning="Saude solicitada",
        selected_agents=["health_monitor"],
        urgency="low",
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    state = {
        "messages": [HumanMessage(content="Como estao minhas campanhas?")],
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
        await supervisor_node(state, config)

    call_args = mock_structured.ainvoke.call_args[0][0]
    # System prompt + 1 HumanMessage (sem reports extra)
    assert len(call_args) == 2
    assert isinstance(call_args[0], SystemMessage)
    assert isinstance(call_args[1], HumanMessage)
    # Verificar que nao tem "Analises ja realizadas" no system prompt
    system_msgs = [m for m in call_args if isinstance(m, SystemMessage)]
    assert len(system_msgs) == 1


# --- Testes unitarios puros das funcoes auxiliares ---


class TestBuildContextMessages:
    """Testes unitarios para _build_context_messages."""

    def test_filters_tool_messages(self):
        """ToolMessage deve ser filtrada do historico."""
        messages = [
            HumanMessage(content="Analise campanhas"),
            AIMessage(content="Vou analisar."),
            ToolMessage(content="resultado da tool", tool_call_id="tc1"),
            HumanMessage(content="E as previsoes?"),
        ]
        result = _build_context_messages(messages, "system")
        # System + 2 Human + 1 AI = 4 (ToolMessage excluida)
        assert len(result) == 4
        assert isinstance(result[0], SystemMessage)
        assert not any(isinstance(m, ToolMessage) for m in result)

    def test_filters_empty_ai_messages(self):
        """AIMessage com content vazio deve ser filtrada."""
        messages = [
            HumanMessage(content="Oi"),
            AIMessage(content=""),
            AIMessage(content="Resposta real"),
            HumanMessage(content="Obrigado"),
        ]
        result = _build_context_messages(messages, "system")
        # System + 2 Human + 1 AI (a vazia foi excluida) = 4
        assert len(result) == 4
        contents = [m.content for m in result[1:]]
        assert "" not in contents

    def test_applies_sliding_window(self):
        """Com mais mensagens que o limite, pega apenas as ultimas."""
        messages = [HumanMessage(content=f"msg-{i}") for i in range(30)]
        with patch("projects.agent.graph.supervisor.MAX_HISTORY_MESSAGES", 10):
            # Re-importar para pegar o valor patchado? Nao — a funcao le a global diretamente
            result = _build_context_messages(messages, "system")
        # System + 10 ultimas = 11
        assert len(result) == 11
        assert result[1].content == "msg-20"
        assert result[-1].content == "msg-29"

    def test_preserves_all_when_under_limit(self):
        """Com menos mensagens que o limite, preserva todas."""
        messages = [
            HumanMessage(content="primeira"),
            AIMessage(content="resposta"),
            HumanMessage(content="segunda"),
        ]
        result = _build_context_messages(messages, "prompt")
        assert len(result) == 4  # System + 3
        assert result[0].content == "prompt"
        assert result[1].content == "primeira"

    def test_empty_messages(self):
        """Com lista vazia, retorna apenas SystemMessage."""
        result = _build_context_messages([], "system")
        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)

    def test_mixed_messages_filters_correctly(self):
        """Historico real com Human, AI, Tool intercalados filtra corretamente."""
        messages = [
            HumanMessage(content="Como esta minha campanha?"),
            AIMessage(content=""),  # vazia — filtrar
            ToolMessage(content="tool result", tool_call_id="tc1"),  # filtrar
            AIMessage(content="Sua campanha esta com CPL de R$15."),
            HumanMessage(content="E as previsoes desses dados?"),
        ]
        result = _build_context_messages(messages, "sys")
        assert len(result) == 4  # System + 2 Human + 1 AI
        types = [type(m).__name__ for m in result]
        assert types == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"]


class TestBuildReportsContext:
    """Testes unitarios para _build_reports_context."""

    def test_empty_reports_returns_empty_string(self):
        """Sem reports, retorna string vazia."""
        assert _build_reports_context([]) == ""

    def test_single_report_formatting(self):
        """Um report deve gerar linha formatada com agent_id, status e summary."""
        reports = [{
            "agent_id": "health_monitor",
            "status": "completed",
            "summary": "Detectadas 3 anomalias",
            "data": None,
            "confidence": 0.9,
        }]
        result = _build_reports_context(reports)
        assert "## Analises ja realizadas" in result
        assert "health_monitor [COMPLETED]" in result
        assert "Detectadas 3 anomalias" in result

    def test_multiple_reports(self):
        """Multiplos reports geram uma linha por report."""
        reports = [
            {"agent_id": "health_monitor", "status": "completed", "summary": "Saude OK", "data": None, "confidence": 0.8},
            {"agent_id": "performance_analyst", "status": "error", "summary": "Timeout na API", "data": None, "confidence": 0.0},
        ]
        result = _build_reports_context(reports)
        assert "health_monitor [COMPLETED]" in result
        assert "performance_analyst [ERROR]" in result

    def test_truncates_long_summary(self):
        """Summary com mais de 200 chars deve ser truncado."""
        long_summary = "x" * 300
        reports = [{"agent_id": "test", "status": "completed", "summary": long_summary, "data": None, "confidence": 1.0}]
        result = _build_reports_context(reports)
        # Nao deve conter o summary completo de 300 chars
        assert long_summary not in result
        # Deve conter os primeiros 200
        assert "x" * 200 in result

    def test_none_summary_handled(self):
        """Report com summary=None nao deve quebrar."""
        reports = [{"agent_id": "test", "status": "completed", "summary": None, "data": None, "confidence": 1.0}]
        result = _build_reports_context(reports)
        assert "test [COMPLETED]:" in result

    def test_missing_fields_use_defaults(self):
        """Report com campos faltando usa defaults seguros."""
        reports = [{}]
        result = _build_reports_context(reports)
        assert "unknown [?]:" in result


# --- Testes de contexto multi-turn para agentes (P1-3) ---


@pytest.mark.asyncio
async def test_supervisor_sends_recent_messages_to_agents():
    """Send() deve conter ultimas N mensagens (nao apenas a ultima)."""
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = RoutingDecision(
        reasoning="Previsoes solicitadas",
        selected_agents=["forecast_scientist"],
        urgency="medium",
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    messages = [
        HumanMessage(content="Analise campanhas c1 e c2"),
        AIMessage(content="Campanhas analisadas. CPL estavel."),
        HumanMessage(content="Agora faca previsoes"),
    ]
    state = {
        "messages": messages,
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
    send_arg = result.goto[0].arg
    # Agente recebe 3 mensagens (todas — menor que o limite de 5)
    assert len(send_arg["messages"]) == 3
    assert send_arg["messages"][0].content == "Analise campanhas c1 e c2"
    assert send_arg["messages"][-1].content == "Agora faca previsoes"


@pytest.mark.asyncio
async def test_supervisor_limits_agent_context_to_n():
    """Com muitas mensagens, Send() contem apenas as ultimas agent_context_messages."""
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = RoutingDecision(
        reasoning="Analise solicitada",
        selected_agents=["health_monitor"],
        urgency="low",
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    # 20 mensagens alternando Human/AI
    messages = []
    for i in range(10):
        messages.append(HumanMessage(content=f"Pergunta {i}"))
        messages.append(AIMessage(content=f"Resposta {i}"))

    state = {
        "messages": messages,
        "user_context": {"user_id": "u1", "account_id": "a1", "account_name": "", "timezone": "America/Sao_Paulo"},
        "agent_reports": [],
        "routing_decision": None,
        "pending_actions": [],
        "synthesis": None,
        "remaining_steps": 100,
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.graph.supervisor.get_model", return_value=mock_model), \
         patch("projects.agent.graph.supervisor.agent_settings") as mock_agent_settings:
        mock_agent_settings.agent_context_messages = 3
        from projects.agent.graph.supervisor import supervisor_node
        result = await supervisor_node(state, config)

    send_arg = result.goto[0].arg
    # Agente recebe no maximo 3 mensagens
    assert len(send_arg["messages"]) == 3
    # Ultima mensagem e a mais recente do historico
    assert send_arg["messages"][-1].content == "Resposta 9"


@pytest.mark.asyncio
async def test_supervisor_single_message_sends_just_one():
    """Com apenas 1 mensagem, Send() contem apenas essa mensagem."""
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = RoutingDecision(
        reasoning="Saude solicitada",
        selected_agents=["health_monitor"],
        urgency="low",
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    state = {
        "messages": [HumanMessage(content="Como estao minhas campanhas?")],
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
    send_arg = result.goto[0].arg
    assert len(send_arg["messages"]) == 1
    assert send_arg["messages"][0].content == "Como estao minhas campanhas?"
