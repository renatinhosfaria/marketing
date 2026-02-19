"""
Testes do schema de roteamento (RoutingDecision + AnalysisScope).

Testa:
  - RoutingDecision: validacao de campos e tipos
  - AnalysisScope: defaults, limites e validacao
  - AgentType literals
  - Validacoes Pydantic (limites de lookback_days, top_n, max_length)
"""

import pytest
from pydantic import ValidationError

from projects.agent.graph.routing import (
    RoutingDecision,
    AnalysisScope,
    ActionDecision,
)


@pytest.mark.asyncio
async def test_routing_decision_valid():
    """RoutingDecision com campos validos."""
    decision = RoutingDecision(
        reasoning="CPL elevado indica anomalia nas campanhas",
        selected_agents=["health_monitor", "performance_analyst"],
        urgency="high",
    )

    assert decision.reasoning == "CPL elevado indica anomalia nas campanhas"
    assert len(decision.selected_agents) == 2
    assert decision.urgency == "high"
    assert decision.scope is not None  # AnalysisScope default


@pytest.mark.asyncio
async def test_routing_decision_default_scope():
    """RoutingDecision sem scope explicito usa AnalysisScope defaults."""
    decision = RoutingDecision(
        reasoning="Pergunta generica",
        selected_agents=["performance_analyst"],
        urgency="low",
    )

    assert decision.scope.entity_type == "campaign"
    assert decision.scope.entity_ids is None
    assert decision.scope.lookback_days == 7
    assert decision.scope.top_n == 10


@pytest.mark.asyncio
async def test_routing_decision_custom_scope():
    """RoutingDecision com scope customizado."""
    scope = AnalysisScope(
        entity_type="adset",
        entity_ids=["adset_1", "adset_2"],
        lookback_days=30,
        top_n=20,
    )
    decision = RoutingDecision(
        reasoning="Analise de adsets",
        selected_agents=["audience_specialist"],
        urgency="medium",
        scope=scope,
    )

    assert decision.scope.entity_type == "adset"
    assert decision.scope.entity_ids == ["adset_1", "adset_2"]
    assert decision.scope.lookback_days == 30


@pytest.mark.asyncio
async def test_analysis_scope_lookback_limits():
    """AnalysisScope deve respeitar limites de lookback_days (1-90)."""
    # Minimo valido
    scope_min = AnalysisScope(lookback_days=1)
    assert scope_min.lookback_days == 1

    # Maximo valido
    scope_max = AnalysisScope(lookback_days=90)
    assert scope_max.lookback_days == 90

    # Abaixo do minimo
    with pytest.raises(ValidationError):
        AnalysisScope(lookback_days=0)

    # Acima do maximo
    with pytest.raises(ValidationError):
        AnalysisScope(lookback_days=91)


@pytest.mark.asyncio
async def test_analysis_scope_top_n_limits():
    """AnalysisScope deve respeitar limites de top_n (1-50)."""
    with pytest.raises(ValidationError):
        AnalysisScope(top_n=0)

    with pytest.raises(ValidationError):
        AnalysisScope(top_n=51)

    valid = AnalysisScope(top_n=50)
    assert valid.top_n == 50


@pytest.mark.asyncio
async def test_routing_decision_empty_agents():
    """RoutingDecision com lista vazia de agentes e valido."""
    decision = RoutingDecision(
        reasoning="Pergunta nao relacionada a ads",
        selected_agents=[],
        urgency="low",
    )

    assert decision.selected_agents == []


@pytest.mark.asyncio
async def test_routing_decision_invalid_urgency():
    """RoutingDecision com urgency invalida deve falhar."""
    with pytest.raises(ValidationError):
        RoutingDecision(
            reasoning="Teste",
            selected_agents=["health_monitor"],
            urgency="critico",  # Nao e um literal valido
        )


@pytest.mark.asyncio
async def test_analysis_scope_model_dump():
    """AnalysisScope.model_dump() deve retornar dict serializavel."""
    scope = AnalysisScope(
        entity_type="campaign",
        entity_ids=["c1"],
        lookback_days=14,
        top_n=5,
    )

    dump = scope.model_dump()
    assert isinstance(dump, dict)
    assert dump["entity_type"] == "campaign"
    assert dump["entity_ids"] == ["c1"]
    assert dump["lookback_days"] == 14
    assert dump["top_n"] == 5


# --- ActionDecision ---


@pytest.mark.asyncio
async def test_action_decision_no_action():
    """ActionDecision com action_needed=False e valido sem outros campos."""
    decision = ActionDecision(
        action_needed=False,
        reason="Nenhuma acao necessaria",
    )
    assert decision.action_needed is False
    assert decision.action_type is None
    assert decision.campaign_id is None
    assert decision.new_value is None


@pytest.mark.asyncio
async def test_action_decision_budget_change_valid():
    """ActionDecision valido para budget_change."""
    decision = ActionDecision(
        action_needed=True,
        action_type="budget_change",
        campaign_id="c1",
        new_value="75.0",
        reason="CPL alto",
    )
    assert decision.action_type == "budget_change"
    assert decision.new_value == "75.0"


@pytest.mark.asyncio
async def test_action_decision_budget_change_non_numeric():
    """ActionDecision rejeita new_value nao numerico para budget_change."""
    with pytest.raises(ValidationError, match="numerico"):
        ActionDecision(
            action_needed=True,
            action_type="budget_change",
            campaign_id="c1",
            new_value="setenta e cinco",
            reason="Teste",
        )


@pytest.mark.asyncio
async def test_action_decision_status_change_valid():
    """ActionDecision valido para status_change."""
    decision = ActionDecision(
        action_needed=True,
        action_type="status_change",
        campaign_id="c1",
        new_value="PAUSED",
        reason="CPL critico",
    )
    assert decision.new_value == "PAUSED"


@pytest.mark.asyncio
async def test_action_decision_status_change_invalid_value():
    """ActionDecision rejeita status invalido para status_change."""
    with pytest.raises(ValidationError, match="ACTIVE ou PAUSED"):
        ActionDecision(
            action_needed=True,
            action_type="status_change",
            campaign_id="c1",
            new_value="DELETED",
            reason="Teste",
        )
