"""
Testes do no Synthesizer — fan-in dos reports + resposta final.

Testa:
  - Sem reports -> fallback message
  - Todos ok -> sintese com LLM
  - Mistura ok + erro -> graceful degradation
  - Prompt construido corretamente
  - Retorno correto (messages + synthesis)
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from langchain_core.messages import HumanMessage, AIMessage


@pytest.mark.asyncio
async def test_synthesizer_no_reports_returns_fallback():
    """Sem reports, synthesizer retorna mensagem de fallback."""
    mock_store = AsyncMock()

    state = {
        "messages": [HumanMessage(content="Como estao minhas campanhas?")],
        "agent_reports": [],
        "user_context": {"user_id": "u1", "account_id": "a1", "account_name": "", "timezone": "America/Sao_Paulo"},
        "routing_decision": None,
        "pending_actions": [],
        "synthesis": None,
        "remaining_steps": 100,
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    from projects.agent.graph.synthesizer import synthesizer_node
    result = await synthesizer_node(state, config, store=mock_store)

    assert "messages" in result
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    assert "desculpe" in msg.content.lower()


@pytest.mark.asyncio
async def test_synthesizer_all_successful_reports():
    """Com todos os reports ok, sintetiza usando LLM."""
    mock_response = AIMessage(content="Suas campanhas estao otimas! CPL estavel.")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response
    mock_store = AsyncMock()

    state = {
        "messages": [HumanMessage(content="Como estao minhas campanhas?")],
        "agent_reports": [
            {
                "agent_id": "health_monitor",
                "status": "completed",
                "summary": "Nenhuma anomalia detectada.",
                "data": {"anomaly_count": 0},
                "confidence": 0.90,
            },
            {
                "agent_id": "performance_analyst",
                "status": "completed",
                "summary": "CPL estavel em R$25.",
                "data": {"metrics_summary": {}},
                "confidence": 0.85,
            },
        ],
        "user_context": {"user_id": "u1", "account_id": "a1", "account_name": "", "timezone": "America/Sao_Paulo"},
        "routing_decision": None,
        "pending_actions": [],
        "synthesis": None,
        "remaining_steps": 100,
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.graph.synthesizer.get_model", return_value=mock_model), \
         patch("projects.agent.graph.synthesizer._maybe_generate_title", new_callable=AsyncMock):
        from projects.agent.graph.synthesizer import synthesizer_node
        result = await synthesizer_node(state, config, store=mock_store)

    assert "messages" in result
    assert "synthesis" in result
    assert result["synthesis"] == mock_response.content
    mock_model.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_synthesizer_mixed_success_and_error():
    """Com mix de sucesso e erro, sintetiza com graceful degradation."""
    mock_response = AIMessage(
        content="Performance estavel. Houve problemas com o monitor de saude."
    )
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response
    mock_store = AsyncMock()

    state = {
        "messages": [HumanMessage(content="Analise completa")],
        "agent_reports": [
            {
                "agent_id": "performance_analyst",
                "status": "completed",
                "summary": "CPL em R$25, CTR 2.1%.",
                "data": {},
                "confidence": 0.85,
            },
            {
                "agent_id": "health_monitor",
                "status": "error",
                "summary": "Agente health_monitor falhou: TimeoutError",
                "data": {"error_code": "SUBGRAPH_EXCEPTION"},
                "confidence": 0.0,
            },
        ],
        "user_context": {"user_id": "u1", "account_id": "a1", "account_name": "", "timezone": "America/Sao_Paulo"},
        "routing_decision": None,
        "pending_actions": [],
        "synthesis": None,
        "remaining_steps": 100,
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.graph.synthesizer.get_model", return_value=mock_model), \
         patch("projects.agent.graph.synthesizer._maybe_generate_title", new_callable=AsyncMock):
        from projects.agent.graph.synthesizer import synthesizer_node
        result = await synthesizer_node(state, config, store=mock_store)

    # Verifica que o prompt inclui agentes com falha
    call_args = mock_model.ainvoke.call_args[0][0]
    # call_args e uma lista de messages; extrair conteudo do system prompt
    prompt_text = " ".join(m.content for m in call_args)
    assert "health_monitor" in prompt_text
    assert "falha" in prompt_text.lower() or "problemas" in prompt_text.lower()
    assert result["synthesis"] == mock_response.content


@pytest.mark.asyncio
async def test_synthesizer_all_errors():
    """Com todos os reports com erro, sintetiza mencionando problemas."""
    mock_response = AIMessage(content="Houve problemas ao analisar seus dados.")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response
    mock_store = AsyncMock()

    state = {
        "messages": [HumanMessage(content="Analise minhas campanhas")],
        "agent_reports": [
            {
                "agent_id": "health_monitor",
                "status": "error",
                "summary": "Falha na ML API",
                "data": None,
                "confidence": 0.0,
            },
            {
                "agent_id": "performance_analyst",
                "status": "error",
                "summary": "Timeout no banco",
                "data": None,
                "confidence": 0.0,
            },
        ],
        "user_context": {"user_id": "u1", "account_id": "a1", "account_name": "", "timezone": "America/Sao_Paulo"},
        "routing_decision": None,
        "pending_actions": [],
        "synthesis": None,
        "remaining_steps": 100,
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.graph.synthesizer.get_model", return_value=mock_model), \
         patch("projects.agent.graph.synthesizer._maybe_generate_title", new_callable=AsyncMock):
        from projects.agent.graph.synthesizer import synthesizer_node
        await synthesizer_node(state, config, store=mock_store)

    # Prompt deve mencionar falhas
    call_args = mock_model.ainvoke.call_args[0][0]
    prompt_text = " ".join(m.content for m in call_args)
    assert "health_monitor" in prompt_text
    assert "performance_analyst" in prompt_text


@pytest.mark.asyncio
async def test_build_synthesis_prompt_format():
    """Verifica formato do prompt de sintese."""
    from projects.agent.graph.synthesizer import _build_synthesis_prompt

    successful = [
        {
            "agent_id": "health_monitor",
            "status": "completed",
            "summary": "Nenhuma anomalia.",
            "data": {},
            "confidence": 0.90,
        },
    ]
    failed = [
        {
            "agent_id": "forecast_scientist",
            "status": "error",
            "summary": "Sem dados",
            "data": None,
            "confidence": 0.0,
        },
    ]

    prompt = _build_synthesis_prompt(successful, failed, "Como esta minha conta?")

    assert "Como esta minha conta?" in prompt
    assert "health_monitor" in prompt
    assert "90%" in prompt
    assert "forecast_scientist" in prompt
    assert "falha" in prompt.lower() or "problemas" in prompt.lower()
    assert "portugues" in prompt.lower()
