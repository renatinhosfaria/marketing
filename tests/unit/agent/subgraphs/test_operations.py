"""
Testes dos nos do subgraph Operations Manager.

Testa:
  - propose_action_node: LLM decide nenhuma acao → retorna None
  - propose_action_node: LLM decide budget change → prepare → interrupt → approved
  - propose_action_node: LLM decide status change → prepare → interrupt → rejected
  - execute_action_node: acao rejeitada gera relatorio
  - execute_action_node: sem acao gera relatorio generico
  - execute_action_node: aprovada chama execute_budget_change
  - execute_action_node: aprovada chama execute_status_change
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from langchain_core.messages import HumanMessage, AIMessage

from projects.agent.subgraphs.operations_manager.nodes import (
    propose_action_node,
    execute_action_node,
)

# Prefixo para patches no namespace do modulo
_NODES = "projects.agent.subgraphs.operations_manager.nodes"


def _make_writer_mock():
    writer = MagicMock()
    return writer


def _make_action_decision(*, action_needed, action_type=None, campaign_id=None, new_value=None, reason="Teste"):
    """Cria mock de ActionDecision."""
    decision = MagicMock()
    decision.action_needed = action_needed
    decision.action_type = action_type
    decision.campaign_id = campaign_id
    decision.new_value = new_value
    decision.reason = reason
    return decision


@pytest.mark.asyncio
async def test_propose_action_node_llm_no_action():
    """propose_action_node: LLM decide que nenhuma acao e necessaria."""
    mock_recs = {"recommendations": [{"title": "Aumentar budget da campanha c1"}]}

    mock_get_recs = AsyncMock()
    mock_get_recs.ainvoke.return_value = {"ok": True, "data": mock_recs, "error": None}

    decision = _make_action_decision(action_needed=False, reason="Sem necessidade")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = decision
    mock_model_raw = MagicMock()
    mock_model_raw.with_structured_output.return_value = mock_model

    mock_store = AsyncMock()

    state = {
        "messages": [HumanMessage(content="Otimize minha campanha")],
        "scope": {"entity_type": "campaign"},
        "proposed_action": None,
        "approval_status": None,
        "execution_result": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch(f"{_NODES}.get_stream_writer", return_value=_make_writer_mock()), \
         patch(f"{_NODES}.get_recommendations", mock_get_recs), \
         patch(f"{_NODES}.get_model", return_value=mock_model_raw):
        result = await propose_action_node(state, config, store=mock_store)

    assert result["proposed_action"] is None
    assert result["approval_status"] is None


@pytest.mark.asyncio
async def test_propose_action_node_budget_action():
    """propose_action_node: LLM decide budget change → prepare → interrupt → approved."""
    mock_recs = {"recommendations": []}
    mock_get_recs = AsyncMock()
    mock_get_recs.ainvoke.return_value = {"ok": True, "data": mock_recs, "error": None}

    decision = _make_action_decision(
        action_needed=True,
        action_type="budget_change",
        campaign_id="c1",
        new_value="75.0",
        reason="CPL alto",
    )
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = decision
    mock_model_raw = MagicMock()
    mock_model_raw.with_structured_output.return_value = mock_model

    mock_prepare = AsyncMock()
    mock_prepare.ainvoke.return_value = {
        "ok": True,
        "data": {
            "action_type": "budget_change",
            "campaign_id": "c1",
            "current_value": 50.0,
            "new_value": 75.0,
            "diff_pct": "+50.0%",
            "reason": "CPL alto",
            "idempotency_key": "abc123",
        },
        "error": None,
    }

    mock_store = AsyncMock()
    mock_approval = {"approved": True, "approval_token": ""}

    state = {
        "messages": [HumanMessage(content="Aumente o budget")],
        "scope": {"entity_type": "campaign"},
        "proposed_action": None,
        "approval_status": None,
        "execution_result": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch(f"{_NODES}.get_stream_writer", return_value=_make_writer_mock()), \
         patch(f"{_NODES}.get_recommendations", mock_get_recs), \
         patch(f"{_NODES}.get_model", return_value=mock_model_raw), \
         patch(f"{_NODES}.prepare_budget_change", mock_prepare), \
         patch(f"{_NODES}.interrupt", return_value=mock_approval), \
         patch(f"{_NODES}.build_approval_token", return_value=""), \
         patch(f"{_NODES}.secrets") as mock_secrets:
        mock_secrets.compare_digest.return_value = True
        result = await propose_action_node(state, config, store=mock_store)

    assert result["approval_status"] == "approved"
    assert result["proposed_action"]["data"]["campaign_id"] == "c1"
    mock_prepare.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_propose_action_node_status_action_rejected():
    """propose_action_node: LLM decide status change → prepare → interrupt → rejected."""
    mock_recs = {"recommendations": []}
    mock_get_recs = AsyncMock()
    mock_get_recs.ainvoke.return_value = {"ok": True, "data": mock_recs, "error": None}

    decision = _make_action_decision(
        action_needed=True,
        action_type="status_change",
        campaign_id="c2",
        new_value="PAUSED",
        reason="CPL critico",
    )
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = decision
    mock_model_raw = MagicMock()
    mock_model_raw.with_structured_output.return_value = mock_model

    mock_prepare = AsyncMock()
    mock_prepare.ainvoke.return_value = {
        "ok": True,
        "data": {
            "action_type": "status_change",
            "campaign_id": "c2",
            "new_status": "PAUSED",
            "reason": "CPL critico",
            "idempotency_key": "def456",
        },
        "error": None,
    }

    mock_store = AsyncMock()
    mock_approval = {"approved": False, "approval_token": ""}

    state = {
        "messages": [HumanMessage(content="Pause campanha c2")],
        "scope": {"entity_type": "campaign"},
        "proposed_action": None,
        "approval_status": None,
        "execution_result": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch(f"{_NODES}.get_stream_writer", return_value=_make_writer_mock()), \
         patch(f"{_NODES}.get_recommendations", mock_get_recs), \
         patch(f"{_NODES}.get_model", return_value=mock_model_raw), \
         patch(f"{_NODES}.prepare_status_change", mock_prepare), \
         patch(f"{_NODES}.interrupt", return_value=mock_approval), \
         patch(f"{_NODES}.build_approval_token", return_value=""), \
         patch(f"{_NODES}.secrets") as mock_secrets:
        mock_secrets.compare_digest.return_value = True
        result = await propose_action_node(state, config, store=mock_store)

    assert result["approval_status"] == "rejected"


@pytest.mark.asyncio
async def test_propose_action_node_prepare_fails():
    """propose_action_node: prepare_* falha → retorna sem interrupt."""
    mock_recs = {"recommendations": []}
    mock_get_recs = AsyncMock()
    mock_get_recs.ainvoke.return_value = {"ok": True, "data": mock_recs, "error": None}

    decision = _make_action_decision(
        action_needed=True,
        action_type="budget_change",
        campaign_id="c1",
        new_value="200.0",
        reason="Teste",
    )
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = decision
    mock_model_raw = MagicMock()
    mock_model_raw.with_structured_output.return_value = mock_model

    mock_prepare = AsyncMock()
    mock_prepare.ainvoke.return_value = {
        "ok": False,
        "data": None,
        "error": {"code": "VALIDATION_ERROR", "message": "Variacao excede limite"},
    }

    mock_store = AsyncMock()

    state = {
        "messages": [HumanMessage(content="Aumente budget")],
        "scope": {"entity_type": "campaign"},
        "proposed_action": None,
        "approval_status": None,
        "execution_result": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch(f"{_NODES}.get_stream_writer", return_value=_make_writer_mock()), \
         patch(f"{_NODES}.get_recommendations", mock_get_recs), \
         patch(f"{_NODES}.get_model", return_value=mock_model_raw), \
         patch(f"{_NODES}.prepare_budget_change", mock_prepare):
        result = await propose_action_node(state, config, store=mock_store)

    assert result["approval_status"] is None
    assert result["proposed_action"]["ok"] is False


@pytest.mark.asyncio
async def test_execute_action_node_rejected():
    """execute_action_node com acao rejeitada gera relatorio informativo."""
    mock_response = AIMessage(content="Recomendacoes disponiveis para analise.")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response
    mock_store = AsyncMock()

    state = {
        "messages": [HumanMessage(content="Otimize campanha")],
        "scope": {},
        "proposed_action": None,
        "approval_status": "rejected",
        "execution_result": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch(f"{_NODES}.get_stream_writer", return_value=_make_writer_mock()), \
         patch(f"{_NODES}.get_model", return_value=mock_model):
        result = await execute_action_node(state, config, store=mock_store)

    report = result["agent_reports"][0]
    assert report["agent_id"] == "operations_manager"
    assert report["status"] == "completed"
    assert report["data"]["action_executed"] is False
    assert "cancelada" in report["summary"].lower()


@pytest.mark.asyncio
async def test_execute_action_node_no_action():
    """execute_action_node sem acao proposta gera relatorio generico."""
    mock_response = AIMessage(content="Nenhuma acao necessaria.")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response
    mock_store = AsyncMock()

    state = {
        "messages": [HumanMessage(content="Como otimizar?")],
        "scope": {},
        "proposed_action": None,
        "approval_status": None,
        "execution_result": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch(f"{_NODES}.get_stream_writer", return_value=_make_writer_mock()), \
         patch(f"{_NODES}.get_model", return_value=mock_model):
        result = await execute_action_node(state, config, store=mock_store)

    report = result["agent_reports"][0]
    assert report["data"]["action_executed"] is False
    assert report["confidence"] == 1.0


@pytest.mark.asyncio
async def test_execute_action_node_calls_execute_budget_change():
    """execute_action_node aprovado chama execute_budget_change."""
    mock_execute = AsyncMock()
    mock_execute.ainvoke.return_value = {
        "ok": True,
        "data": {"message": "Orcamento atualizado: R$50.0 -> R$75.0"},
        "error": None,
    }
    mock_store = AsyncMock()

    state = {
        "messages": [HumanMessage(content="Confirme alteracao")],
        "scope": {},
        "proposed_action": {
            "ok": True,
            "data": {
                "action_type": "budget_change",
                "campaign_id": "c1",
                "current_value": 50.0,
                "new_value": 75.0,
                "idempotency_key": "abc123",
            },
        },
        "approval_status": "approved",
        "execution_result": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch(f"{_NODES}.get_stream_writer", return_value=_make_writer_mock()), \
         patch(f"{_NODES}.execute_budget_change", mock_execute):
        result = await execute_action_node(state, config, store=mock_store)

    report = result["agent_reports"][0]
    assert report["agent_id"] == "operations_manager"
    assert report["data"]["action_executed"] is True
    assert "atualizado" in report["summary"].lower()
    mock_execute.ainvoke.assert_called_once()
    call_args = mock_execute.ainvoke.call_args[0][0]
    assert call_args["campaign_id"] == "c1"
    assert call_args["new_daily_budget"] == 75.0
    assert call_args["idempotency_key"] == "abc123"


@pytest.mark.asyncio
async def test_execute_action_node_applies_budget_override_canonical_field():
    """execute_action_node aplica new_budget_override aprovado antes de executar."""
    mock_execute = AsyncMock()
    mock_execute.ainvoke.return_value = {
        "ok": True,
        "data": {"message": "Orcamento atualizado"},
        "error": None,
    }
    mock_store = AsyncMock()

    state = {
        "messages": [HumanMessage(content="Confirme alteracao")],
        "scope": {},
        "proposed_action": {
            "ok": True,
            "data": {
                "action_type": "budget_change",
                "campaign_id": "c1",
                "current_value": 50.0,
                "new_value": 75.0,
                "idempotency_key": "abc123",
            },
            "approved_overrides": {
                "approved": True,
                "approval_token": "token",
                "new_budget_override": 62.5,
            },
        },
        "approval_status": "approved",
        "execution_result": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch(f"{_NODES}.get_stream_writer", return_value=_make_writer_mock()), \
         patch(f"{_NODES}.execute_budget_change", mock_execute):
        await execute_action_node(state, config, store=mock_store)

    call_args = mock_execute.ainvoke.call_args[0][0]
    assert call_args["new_daily_budget"] == 62.5


@pytest.mark.asyncio
async def test_execute_action_node_applies_budget_override_legacy_field():
    """execute_action_node aplica override_value legado quando presente."""
    mock_execute = AsyncMock()
    mock_execute.ainvoke.return_value = {
        "ok": True,
        "data": {"message": "Orcamento atualizado"},
        "error": None,
    }
    mock_store = AsyncMock()

    state = {
        "messages": [HumanMessage(content="Confirme alteracao")],
        "scope": {},
        "proposed_action": {
            "ok": True,
            "data": {
                "action_type": "budget_change",
                "campaign_id": "c1",
                "current_value": 50.0,
                "new_value": 75.0,
                "idempotency_key": "abc123",
            },
            "approved_overrides": {
                "approved": True,
                "approval_token": "token",
                "override_value": 64.0,
            },
        },
        "approval_status": "approved",
        "execution_result": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch(f"{_NODES}.get_stream_writer", return_value=_make_writer_mock()), \
         patch(f"{_NODES}.execute_budget_change", mock_execute):
        await execute_action_node(state, config, store=mock_store)

    call_args = mock_execute.ainvoke.call_args[0][0]
    assert call_args["new_daily_budget"] == 64.0


@pytest.mark.asyncio
async def test_execute_action_node_calls_execute_status_change():
    """execute_action_node aprovado chama execute_status_change."""
    mock_execute = AsyncMock()
    mock_execute.ainvoke.return_value = {
        "ok": True,
        "data": {"message": "Campanha paused com sucesso."},
        "error": None,
    }
    mock_store = AsyncMock()

    state = {
        "messages": [HumanMessage(content="Pause campanha")],
        "scope": {},
        "proposed_action": {
            "ok": True,
            "data": {
                "action_type": "status_change",
                "campaign_id": "c2",
                "new_status": "PAUSED",
                "idempotency_key": "def456",
            },
        },
        "approval_status": "approved",
        "execution_result": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch(f"{_NODES}.get_stream_writer", return_value=_make_writer_mock()), \
         patch(f"{_NODES}.execute_status_change", mock_execute):
        result = await execute_action_node(state, config, store=mock_store)

    report = result["agent_reports"][0]
    assert report["data"]["action_executed"] is True
    mock_execute.ainvoke.assert_called_once()
    call_args = mock_execute.ainvoke.call_args[0][0]
    assert call_args["campaign_id"] == "c2"
    assert call_args["new_status"] == "PAUSED"


@pytest.mark.asyncio
async def test_execute_action_node_execution_fails():
    """execute_action_node: execucao falha retorna relatorio com erro."""
    mock_execute = AsyncMock()
    mock_execute.ainvoke.return_value = {
        "ok": False,
        "data": None,
        "error": {"code": "FB_API_ERROR", "message": "Token expirado"},
    }
    mock_store = AsyncMock()

    state = {
        "messages": [HumanMessage(content="Confirme")],
        "scope": {},
        "proposed_action": {
            "ok": True,
            "data": {
                "action_type": "budget_change",
                "campaign_id": "c1",
                "new_value": 75.0,
                "idempotency_key": "abc123",
            },
        },
        "approval_status": "approved",
        "execution_result": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch(f"{_NODES}.get_stream_writer", return_value=_make_writer_mock()), \
         patch(f"{_NODES}.execute_budget_change", mock_execute):
        result = await execute_action_node(state, config, store=mock_store)

    report = result["agent_reports"][0]
    assert report["data"]["action_executed"] is False
    assert "falha" in report["summary"].lower()


@pytest.mark.asyncio
async def test_propose_action_node_invalid_approval_token():
    """propose_action_node rejeita quando token de aprovacao nao bate (anti-forgery)."""
    mock_recs = {"recommendations": []}
    mock_get_recs = AsyncMock()
    mock_get_recs.ainvoke.return_value = {"ok": True, "data": mock_recs, "error": None}

    decision = _make_action_decision(
        action_needed=True,
        action_type="budget_change",
        campaign_id="c1",
        new_value="75.0",
        reason="CPL alto",
    )
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = decision
    mock_model_raw = MagicMock()
    mock_model_raw.with_structured_output.return_value = mock_model

    mock_prepare = AsyncMock()
    mock_prepare.ainvoke.return_value = {
        "ok": True,
        "data": {
            "action_type": "budget_change",
            "campaign_id": "c1",
            "current_value": 50.0,
            "new_value": 75.0,
            "idempotency_key": "abc123",
        },
        "error": None,
    }

    mock_store = AsyncMock()
    # Token forjado nao bate com o gerado
    mock_approval = {"approved": True, "approval_token": "forged_token_xyz"}

    state = {
        "messages": [HumanMessage(content="Aumente budget")],
        "scope": {"entity_type": "campaign"},
        "proposed_action": None,
        "approval_status": None,
        "execution_result": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch(f"{_NODES}.get_stream_writer", return_value=_make_writer_mock()), \
         patch(f"{_NODES}.get_recommendations", mock_get_recs), \
         patch(f"{_NODES}.get_model", return_value=mock_model_raw), \
         patch(f"{_NODES}.prepare_budget_change", mock_prepare), \
         patch(f"{_NODES}.interrupt", return_value=mock_approval), \
         patch(f"{_NODES}.build_approval_token", return_value="real_token_abc"):
        # NAO mockamos secrets.compare_digest — usa o real
        result = await propose_action_node(state, config, store=mock_store)

    assert result["approval_status"] == "rejected"
    assert "Token invalido" in result.get("execution_result", {}).get("error", "")


@pytest.mark.asyncio
async def test_propose_action_node_llm_invalid_budget_value():
    """propose_action_node: LLM gera new_value nao numerico → retorna sem proposta."""
    mock_recs = {"recommendations": []}
    mock_get_recs = AsyncMock()
    mock_get_recs.ainvoke.return_value = {"ok": True, "data": mock_recs, "error": None}

    decision = _make_action_decision(
        action_needed=True,
        action_type="budget_change",
        campaign_id="c1",
        new_value="setenta e cinco reais",  # nao numerico
        reason="CPL alto",
    )
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = decision
    mock_model_raw = MagicMock()
    mock_model_raw.with_structured_output.return_value = mock_model

    mock_store = AsyncMock()

    state = {
        "messages": [HumanMessage(content="Aumente budget")],
        "scope": {"entity_type": "campaign"},
        "proposed_action": None,
        "approval_status": None,
        "execution_result": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch(f"{_NODES}.get_stream_writer", return_value=_make_writer_mock()), \
         patch(f"{_NODES}.get_recommendations", mock_get_recs), \
         patch(f"{_NODES}.get_model", return_value=mock_model_raw):
        result = await propose_action_node(state, config, store=mock_store)

    assert result["proposed_action"] is None
    assert result["approval_status"] is None


@pytest.mark.asyncio
async def test_execute_action_node_missing_proposal_data():
    """execute_action_node: dados da proposta incompletos gera relatorio de erro."""
    mock_store = AsyncMock()

    state = {
        "messages": [HumanMessage(content="Confirme")],
        "scope": {},
        "proposed_action": {
            "ok": True,
            "data": {
                "action_type": "budget_change",
                # campaign_id ausente!
            },
        },
        "approval_status": "approved",
        "execution_result": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch(f"{_NODES}.get_stream_writer", return_value=_make_writer_mock()):
        result = await execute_action_node(state, config, store=mock_store)

    report = result["agent_reports"][0]
    assert report["data"]["action_executed"] is False
    assert "incompletos" in report["summary"].lower()
