"""
Tools do Gerente de Operacoes — operacoes de escrita com validacao e idempotencia.

Arquitetura: Tools Puras + Interrupt no Node.
O interrupt() vive nos nodes do subgraph (nao nas tools). Tools sao "puras":
validam, preparam proposta, ou executam.

Tools:
  - prepare_budget_change: valida e prepara proposta de mudanca de orcamento
  - execute_budget_change: executa mudanca apos aprovacao humana
  - prepare_status_change: valida e prepara proposta de mudanca de status
  - execute_status_change: executa mudanca apos aprovacao humana
  - get_recommendations: busca recomendacoes do sistema ML
  - apply_recommendation: prepara aplicacao de recomendacao

Funcoes internas:
  - _validate_write_operation: validacoes obrigatorias para escrita
  - _count_recent_actions: conta acoes recentes (cooldown)
  - _generate_idempotency_key: gera chave de idempotencia estavel
  - build_approval_token: token anti-forgery deterministico
  - _record_action_with_baseline: registra acao com metricas baseline
  - _fetch_baseline_metrics: busca metricas medias para baseline
  - _already_executed: verifica se acao ja foi executada (idempotencia)
"""

import hashlib
import hmac
from datetime import datetime, timezone, timedelta
from typing import Annotated, Literal
from uuid import uuid4

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from sqlalchemy import select, func, text

from shared.db.session import async_session_maker
from shared.db.models import SistemaFacebookAdsInsightsHistory

from projects.agent.config import agent_settings
from projects.agent.tools import fb_service
from projects.agent.tools.result import ToolResult, tool_success, tool_error
from projects.agent.tools.http_client import _ml_api_call
from projects.agent.tools.config_resolver import resolve_config_id
from projects.agent.memory.namespaces import StoreNamespace

import structlog


logger = structlog.get_logger()


# --- Excecao de validacao ---

class WriteValidationError(Exception):
    """Erro de validacao em operacao de escrita."""
    pass


# --- Validacoes duras para operacoes de escrita ---

async def _validate_write_operation(
    campaign_id: str,
    config_id: int,
    operation_type: str,
    new_value: float | str | None = None,
    current_value: float | str | None = None,
) -> None:
    """Validacoes obrigatorias ANTES do interrupt e ANTES da execucao.

    Raises WriteValidationError com mensagem amigavel para o LLM.
    """
    # 1. Ownership: campanha pertence a conta do usuario (query filtra por config_id)
    campaign = await fb_service.get_campaign_by_config(campaign_id, config_id)
    if not campaign:
        raise WriteValidationError(
            f"Campanha {campaign_id} nao encontrada ou nao pertence a esta conta."
        )

    # 2. Valores minimos/maximos (validar ANTES de calcular variacao)
    if operation_type == "budget_change" and new_value is not None:
        new_val = float(new_value)
        if new_val <= 0:
            raise WriteValidationError("Orcamento deve ser um valor positivo.")
        if new_val < 1.0:
            raise WriteValidationError("Orcamento minimo e R$1.00.")
        if new_val > 100_000.0:
            raise WriteValidationError("Orcamento maximo e R$100.000,00.")

    # 3. Limites de variacao (para budget)
    if operation_type == "budget_change" and current_value and new_value:
        current_val = float(current_value)
        new_val = float(new_value)
        if current_val > 0:
            diff_pct = abs((new_val - current_val) / current_val) * 100
            if diff_pct > agent_settings.max_budget_change_pct:
                raise WriteValidationError(
                    f"Variacao de {diff_pct:.1f}% excede limite de "
                    f"{agent_settings.max_budget_change_pct}%. "
                    f"Reduza o valor ou altere em etapas."
                )

    # 4. Cooldown: nao permitir N alteracoes em M minutos
    recent_actions = await _count_recent_actions(
        campaign_id, operation_type, minutes=10,
    )
    if recent_actions >= 3:
        raise WriteValidationError(
            "Limite de alteracoes atingido (3 em 10 minutos). Aguarde."
        )


async def _count_recent_actions(
    campaign_id: str,
    operation_type: str,
    minutes: int,
) -> int:
    """Conta acoes recentes via query direta (DB, nao Store).

    Usa tabela agent_action_log (criada na migracao do Agent).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    try:
        async with async_session_maker() as session:
            result = await session.execute(
                text(
                    "SELECT COUNT(*) FROM agent_action_log "
                    "WHERE campaign_id = :cid AND operation_type = :op "
                    "AND status IN ('executed', 'success') "
                    "AND executed_at > :cutoff"
                ),
                {"cid": campaign_id, "op": operation_type, "cutoff": cutoff},
            )
            return result.scalar() or 0
    except Exception:
        # Se a tabela nao existe ainda, retorna 0 (graceful degradation)
        return 0


# --- Idempotencia e anti-forgery ---

def _generate_idempotency_key(
    thread_id: str,
    campaign_id: str,
    operation_type: str,
    value: str,
) -> str:
    """Gera chave de idempotencia estavel por acao planejada.

    Baseada em (thread_id, campaign_id, operation_type, value) — nao depende
    de hora. Mesma acao no mesmo thread sempre gera a mesma key, evitando
    duplicacao por retry/resume mesmo se cruzar virada de hora.
    """
    raw = f"{thread_id}:{campaign_id}:{operation_type}:{value}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_approval_token(thread_id: str, idempotency_key: str) -> str:
    """Token anti-forgery deterministico por proposta aprovada.

    IMPORTANTE: interrupt() reexecuta o codigo anterior no resume. Se o token
    for aleatorio a cada execucao, o resume legitimo falha. Por isso, o token
    precisa ser estavel para a mesma proposta.
    """
    raw = f"{thread_id}:{idempotency_key}".encode()
    secret = agent_settings.approval_token_secret.encode()
    return hmac.new(secret, raw, hashlib.sha256).hexdigest()[:32]


async def _already_executed(idempotency_key: str) -> bool:
    """Verifica se acao ja foi executada (idempotencia).

    Busca na tabela agent_action_log por idempotency_key.
    """
    try:
        async with async_session_maker() as session:
            result = await session.execute(
                text(
                    "SELECT 1 FROM agent_action_log "
                    "WHERE idempotency_key = :key "
                    "AND status IN ('executing', 'executed', 'success') "
                    "LIMIT 1"
                ),
                {"key": idempotency_key},
            )
            return result.scalar() is not None
    except Exception:
        return False


def _to_optional_int(value: str | int | None) -> int | None:
    """Converte valor para int quando possivel (fallback None)."""
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


async def _claim_action_log(
    *,
    idempotency_key: str,
    campaign_id: str,
    operation_type: str,
    config_id: int,
    config: dict,
    details: dict | None = None,
) -> bool:
    """Tenta reservar idempotency_key no agent_action_log.

    Retorna:
      - True: claim obtido (seguir com execucao)
      - False: conflito ON CONFLICT (acao ja em execucao/ja executada)
    """
    payload = dict(details or {})
    payload.update({
        "account_id_raw": config.get("account_id"),
        "user_id_raw": config.get("user_id"),
        "config_id": config_id,
    })
    try:
        async with async_session_maker() as session:
            result = await session.execute(
                text(
                    "INSERT INTO agent_action_log ("
                    "  campaign_id, operation_type, details, idempotency_key, "
                    "  user_id, account_id, status"
                    ") VALUES ("
                    "  :campaign_id, :operation_type, :details, :idempotency_key, "
                    "  :user_id, :account_id, :status"
                    ") "
                    "ON CONFLICT (idempotency_key) DO NOTHING "
                    "RETURNING id"
                ),
                {
                    "campaign_id": campaign_id,
                    "operation_type": operation_type,
                    "details": payload,
                    "idempotency_key": idempotency_key,
                    "user_id": _to_optional_int(config.get("user_id")),
                    "account_id": config_id,
                    "status": "executing",
                },
            )
            inserted = result.scalar() is not None
            await session.commit()
            return inserted
    except Exception as exc:
        # Se log nao estiver disponivel, nao bloqueia a operacao.
        logger.warning(
            "agent_action_log.claim_failed",
            error=str(exc),
            operation_type=operation_type,
        )
        return True


async def _mark_action_log_status(
    *,
    idempotency_key: str,
    status: str,
    details: dict | None = None,
) -> None:
    """Atualiza status final da acao no agent_action_log (best effort)."""
    payload = details or {}
    try:
        async with async_session_maker() as session:
            await session.execute(
                text(
                    "UPDATE agent_action_log "
                    "SET status = :status, "
                    "details = COALESCE(details::jsonb, '{}'::jsonb) || (:details)::jsonb, "
                    "executed_at = :executed_at "
                    "WHERE idempotency_key = :idempotency_key"
                ),
                {
                    "status": status,
                    "details": payload,
                    "executed_at": datetime.now(timezone.utc),
                    "idempotency_key": idempotency_key,
                },
            )
            await session.commit()
    except Exception as exc:
        logger.warning(
            "agent_action_log.mark_status_failed",
            error=str(exc),
            status=status,
        )


# --- Auditoria de Impacto Pos-Acao ---

async def _record_action_with_baseline(
    store: BaseStore,
    cfg: dict,
    campaign_id: str,
    action_type: str,
    details: dict,
) -> None:
    """Registra acao executada COM snapshot de metricas (baseline).

    O store e recebido do node que chama esta funcao.
    O snapshot permite calcular impacto real 7 dias depois.

    Fluxo:
      1. Acao aprovada -> grava action + before_metrics (ultimos 7 dias)
      2. Job Celery (7 dias depois) -> coleta after_metrics e calcula impact_pct
      3. Metrica Prometheus agent_action_impact_pct atualizada
    """
    baseline = await _fetch_baseline_metrics(campaign_id, lookback_days=7)

    action_record = {
        "action_type": action_type,
        "campaign_id": campaign_id,
        "details": details,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "before_metrics": baseline,
        "after_metrics": None,       # Preenchido pelo job de impacto
        "impact_pct": None,          # Preenchido pelo job de impacto
        "impact_calculated_at": None,
    }

    await store.aput(
        StoreNamespace.account_actions(cfg["user_id"], cfg["account_id"]),
        key=details.get("idempotency_key", str(uuid4())),
        value=action_record,
    )


async def _fetch_baseline_metrics(
    campaign_id: str,
    lookback_days: int = 7,
) -> dict:
    """Busca metricas medias dos ultimos N dias para baseline.

    Consulta a tabela sistema_facebook_ads_insights_history para calcular
    medias de CPL, CTR, leads e spend.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    try:
        async with async_session_maker() as session:
            result = await session.execute(
                select(
                    func.avg(SistemaFacebookAdsInsightsHistory.cost_per_lead).label("cpl"),
                    func.avg(SistemaFacebookAdsInsightsHistory.ctr).label("ctr"),
                    func.sum(SistemaFacebookAdsInsightsHistory.leads).label("leads"),
                    func.sum(SistemaFacebookAdsInsightsHistory.spend).label("spend"),
                ).where(
                    SistemaFacebookAdsInsightsHistory.campaign_id == campaign_id,
                    SistemaFacebookAdsInsightsHistory.date >= cutoff.date(),
                )
            )
            row = result.one()
            return {
                "cpl": float(row.cpl or 0),
                "ctr": float(row.ctr or 0),
                "leads": int(row.leads or 0),
                "spend": float(row.spend or 0),
            }
    except Exception:
        return {"cpl": 0, "ctr": 0, "leads": 0, "spend": 0}


# --- Tools PURAS (sem interrupt) ---

@tool
async def prepare_budget_change(
    campaign_id: str,
    new_daily_budget: float,
    reason: str,
    config: RunnableConfig = None,
) -> ToolResult:
    """Valida e prepara proposta de mudanca de orcamento (sem executar).

    Retorna proposta estruturada para o node decidir se interrompe.

    Args:
        campaign_id: ID da campanha no Facebook.
        new_daily_budget: Novo budget diario em reais.
        reason: Motivo da mudanca.
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")
    thread_id = cfg.get("thread_id", "")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    current_budget = await fb_service.get_campaign_budget(campaign_id)

    if current_budget > 0:
        diff_pct = ((new_daily_budget - current_budget) / current_budget) * 100
    else:
        diff_pct = 100.0

    try:
        await _validate_write_operation(
            campaign_id, config_id, "budget_change",
            new_value=new_daily_budget, current_value=current_budget,
        )
    except WriteValidationError as e:
        return tool_error("VALIDATION_ERROR", str(e))

    idempotency_key = _generate_idempotency_key(
        thread_id, campaign_id, "budget", str(new_daily_budget),
    )

    return tool_success({
        "action_type": "budget_change",
        "campaign_id": campaign_id,
        "current_value": current_budget,
        "new_value": new_daily_budget,
        "diff_pct": f"{diff_pct:+.1f}%",
        "reason": reason,
        "idempotency_key": idempotency_key,
    })


@tool
async def execute_budget_change(
    campaign_id: str,
    new_daily_budget: float,
    idempotency_key: str,
    config: RunnableConfig = None,
    store: Annotated[BaseStore, InjectedStore()] = None,
) -> ToolResult:
    """Executa mudanca de orcamento (chamado APOS aprovacao humana).

    Re-valida antes de executar (estado pode ter mudado durante o interrupt).

    Args:
        campaign_id: ID da campanha no Facebook.
        new_daily_budget: Novo budget diario em reais.
        idempotency_key: Chave de idempotencia da proposta.
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    current_budget = await fb_service.get_campaign_budget(campaign_id)

    # Re-validar APOS aprovacao
    try:
        await _validate_write_operation(
            campaign_id, config_id, "budget_change",
            new_value=new_daily_budget, current_value=current_budget,
        )
    except WriteValidationError as e:
        return tool_error(
            "VALIDATION_ERROR",
            f"Operacao bloqueada na re-validacao: {e}",
        )

    if await _already_executed(idempotency_key):
        return tool_success({
            "message": "Operacao ja executada (idempotencia). Nenhuma acao tomada."
        })

    if agent_settings.enable_strict_write_path:
        claimed = await _claim_action_log(
            idempotency_key=idempotency_key,
            campaign_id=campaign_id,
            operation_type="budget_change",
            config_id=config_id,
            config=cfg,
            details={
                "phase": "claimed",
                "campaign_id": campaign_id,
                "new_daily_budget": new_daily_budget,
            },
        )
        if not claimed:
            return tool_success({
                "message": "Operacao ja executada (idempotencia). Nenhuma acao tomada."
            })

    try:
        await fb_service.update_budget(campaign_id, new_daily_budget)
    except Exception as e:
        await _mark_action_log_status(
            idempotency_key=idempotency_key,
            status="failed",
            details={
                "campaign_id": campaign_id,
                "new_daily_budget": new_daily_budget,
                "error": str(e),
            },
        )
        return tool_error(
            "FB_API_ERROR",
            f"Erro ao atualizar orcamento via Facebook API: {e}",
            retryable=True,
        )

    if agent_settings.enable_strict_write_path:
        await _mark_action_log_status(
            idempotency_key=idempotency_key,
            status="executed",
            details={
                "campaign_id": campaign_id,
                "old_daily_budget": current_budget,
                "new_daily_budget": new_daily_budget,
            },
        )

    if store:
        await _record_action_with_baseline(store, cfg, campaign_id, "budget_change", {
            "old": current_budget,
            "new": new_daily_budget,
            "idempotency_key": idempotency_key,
        })

    return tool_success({
        "message": f"Orcamento atualizado: R${current_budget} -> R${new_daily_budget}"
    })


@tool
async def prepare_status_change(
    campaign_id: str,
    new_status: Literal["ACTIVE", "PAUSED"],
    reason: str,
    config: RunnableConfig = None,
) -> ToolResult:
    """Valida e prepara proposta de mudanca de status (sem executar).

    Args:
        campaign_id: ID da campanha no Facebook.
        new_status: Novo status (ACTIVE ou PAUSED).
        reason: Motivo da mudanca.
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")
    thread_id = cfg.get("thread_id", "")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    try:
        await _validate_write_operation(
            campaign_id, config_id, "status_change",
        )
    except WriteValidationError as e:
        return tool_error("VALIDATION_ERROR", str(e))

    idempotency_key = _generate_idempotency_key(
        thread_id, campaign_id, "status", new_status,
    )

    return tool_success({
        "action_type": "status_change",
        "campaign_id": campaign_id,
        "new_status": new_status,
        "reason": reason,
        "idempotency_key": idempotency_key,
    })


@tool
async def execute_status_change(
    campaign_id: str,
    new_status: Literal["ACTIVE", "PAUSED"],
    idempotency_key: str,
    config: RunnableConfig = None,
    store: Annotated[BaseStore, InjectedStore()] = None,
) -> ToolResult:
    """Executa mudanca de status (chamado APOS aprovacao humana).

    Args:
        campaign_id: ID da campanha no Facebook.
        new_status: Novo status (ACTIVE ou PAUSED).
        idempotency_key: Chave de idempotencia da proposta.
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    try:
        await _validate_write_operation(
            campaign_id, config_id, "status_change",
        )
    except WriteValidationError as e:
        return tool_error(
            "VALIDATION_ERROR",
            f"Operacao bloqueada na re-validacao: {e}",
        )

    if await _already_executed(idempotency_key):
        return tool_success({"message": "Operacao ja executada (idempotencia)."})

    if agent_settings.enable_strict_write_path:
        claimed = await _claim_action_log(
            idempotency_key=idempotency_key,
            campaign_id=campaign_id,
            operation_type="status_change",
            config_id=config_id,
            config=cfg,
            details={
                "phase": "claimed",
                "campaign_id": campaign_id,
                "new_status": new_status,
            },
        )
        if not claimed:
            return tool_success({"message": "Operacao ja executada (idempotencia)."})

    try:
        await fb_service.update_status(campaign_id, new_status)
    except Exception as e:
        await _mark_action_log_status(
            idempotency_key=idempotency_key,
            status="failed",
            details={
                "campaign_id": campaign_id,
                "new_status": new_status,
                "error": str(e),
            },
        )
        return tool_error(
            "FB_API_ERROR",
            f"Erro ao atualizar status via Facebook API: {e}",
            retryable=True,
        )

    if agent_settings.enable_strict_write_path:
        await _mark_action_log_status(
            idempotency_key=idempotency_key,
            status="executed",
            details={
                "campaign_id": campaign_id,
                "new_status": new_status,
            },
        )

    if store:
        await _record_action_with_baseline(store, cfg, campaign_id, "status_change", {
            "new_status": new_status,
            "idempotency_key": idempotency_key,
        })

    return tool_success({
        "message": f"Campanha {new_status.lower()} com sucesso."
    })


@tool
async def get_recommendations(
    entity_type: Literal["campaign", "adset", "ad"],
    config: RunnableConfig = None,
) -> ToolResult:
    """Busca recomendacoes geradas pelo sistema ML.

    Args:
        entity_type: Nivel de entidade (campaign, adset, ad).
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    return await _ml_api_call(
        "get",
        "/api/v1/recommendations",
        params={"config_id": config_id, "entity_type": entity_type},
        account_id=account_id,
    )
