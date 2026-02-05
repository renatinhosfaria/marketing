#!/usr/bin/env python3
"""
Script de backfill completo do Facebook Ads.
Sincroniza campanhas, conjuntos, anuncios e insights historicos.

Periodo: 2025-01-01 ate 2026-02-01

Uso:
    python scripts/sync_full_backfill.py
    python scripts/sync_full_backfill.py --config-id 1
    python scripts/sync_full_backfill.py --dry-run
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime, date, timedelta
from decimal import Decimal

# Garantir que o path do projeto esta no sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select, and_, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from shared.core.logging import get_logger
from shared.db.session import isolated_async_session
from shared.db.models.famachat_readonly import (
    SistemaFacebookAdsConfig,
    SistemaFacebookAdsCampaigns,
    SistemaFacebookAdsAdsets,
    SistemaFacebookAdsAds,
    SistemaFacebookAdsInsightsHistory,
    SistemaFacebookAdsInsightsToday,
)
from projects.facebook_ads.services.sync_campaigns import SyncCampaignsService
from projects.facebook_ads.services.sync_adsets_ads import SyncAdSetsAdsService
from projects.facebook_ads.services.sync_insights import SyncInsightsService
from projects.facebook_ads.client.base import FacebookGraphClient
from projects.facebook_ads.client.insights import InsightsClient
from projects.facebook_ads.security.token_encryption import decrypt_token
from projects.facebook_ads.utils.metrics_calculator import (
    calculate_ctr, calculate_cpc, calculate_cpm, calculate_cpl,
    calculate_frequency, extract_leads_from_actions,
)
from projects.facebook_ads.models.sync import SistemaFacebookAdsSyncHistory, SyncStatus

logger = get_logger(__name__)

# === Configuracao do backfill ===
DATE_START = date(2025, 1, 1)
DATE_END = date(2026, 2, 1)
CHUNK_DAYS = 30  # Insights em chunks de 30 dias (evitar timeout e rate limit)
BATCH_SIZE = 500  # Flush a cada N registros


def strip_act_prefix(account_id: str) -> str:
    """Remove prefixo 'act_' se presente (evita duplicacao act_act_)."""
    if account_id and account_id.startswith("act_"):
        return account_id[4:]
    return account_id


def generate_monthly_chunks(start: date, end: date, chunk_days: int = 30) -> list[tuple[str, str]]:
    """Gera intervalos de datas em chunks para evitar timeout na API."""
    chunks = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + timedelta(days=1)
    return chunks


async def get_active_configs(db: AsyncSession, config_id: int = None) -> list:
    """Busca configuracoes ativas do Facebook Ads."""
    query = select(SistemaFacebookAdsConfig).where(
        SistemaFacebookAdsConfig.is_active == True
    )
    if config_id:
        query = query.where(SistemaFacebookAdsConfig.id == config_id)

    result = await db.execute(query)
    configs = result.scalars().all()
    return configs


async def sync_campaigns_adsets_ads(db: AsyncSession, config: SistemaFacebookAdsConfig) -> dict:
    """Sincroniza campanhas, conjuntos de anuncios e anuncios."""
    campaigns_svc = SyncCampaignsService(db)
    adsets_ads_svc = SyncAdSetsAdsService(db)

    results = {}

    # 1. Campanhas
    print(f"\n  [1/3] Sincronizando campanhas...")
    camp_result = await campaigns_svc.sync(config)
    results["campaigns"] = camp_result
    print(f"        Campanhas: {camp_result['synced']} total | "
          f"{camp_result['created']} criadas | {camp_result['updated']} atualizadas | "
          f"{camp_result['errors']} erros")

    # 2. Conjuntos de anuncios
    print(f"  [2/3] Sincronizando conjuntos de anuncios...")
    adsets_result = await adsets_ads_svc.sync_adsets(config)
    results["adsets"] = adsets_result
    print(f"        Ad Sets: {adsets_result['synced']} total | "
          f"{adsets_result['created']} criados | {adsets_result['updated']} atualizados | "
          f"{adsets_result['errors']} erros")

    # 3. Anuncios
    print(f"  [3/3] Sincronizando anuncios...")
    ads_result = await adsets_ads_svc.sync_ads(config)
    results["ads"] = ads_result
    print(f"        Ads: {ads_result['synced']} total | "
          f"{ads_result['created']} criados | {ads_result['updated']} atualizados | "
          f"{ads_result['errors']} erros")

    return results


async def sync_insights_chunk(
    db: AsyncSession,
    config: SistemaFacebookAdsConfig,
    since: str,
    until: str,
) -> dict:
    """Sincroniza insights de um periodo especifico para a tabela history."""
    access_token = decrypt_token(config.access_token)
    graph_client = FacebookGraphClient(access_token, config.account_id)
    insights_client = InsightsClient(graph_client)
    insights_svc = SyncInsightsService(db)

    time_range = {"since": since, "until": until}

    try:
        # Determinar se usa async report (> 30 dias usa async)
        days = (datetime.strptime(until, "%Y-%m-%d") - datetime.strptime(since, "%Y-%m-%d")).days
        if days > 30:
            fb_insights = await insights_client.get_insights_async(
                config.account_id if config.account_id.startswith("act_") else f"act_{config.account_id}",
                time_range=time_range,
                level="ad",
            )
        else:
            fb_insights = await insights_client.get_insights(
                config.account_id if config.account_id.startswith("act_") else f"act_{config.account_id}",
                time_range=time_range,
                level="ad",
            )

        inserted = 0
        updated = 0
        errors = 0

        for insight in fb_insights:
            try:
                insight_date = insight.get("date_start", "")
                ad_id = insight.get("ad_id", "")

                if not insight_date or not ad_id:
                    continue

                # Verificar se ja existe
                existing = await db.execute(
                    select(SistemaFacebookAdsInsightsHistory).where(
                        and_(
                            SistemaFacebookAdsInsightsHistory.config_id == config.id,
                            SistemaFacebookAdsInsightsHistory.ad_id == ad_id,
                            SistemaFacebookAdsInsightsHistory.date == datetime.strptime(insight_date, "%Y-%m-%d"),
                        )
                    )
                )
                existing_obj = existing.scalar_one_or_none()

                if existing_obj:
                    insights_svc._update_insight_history(existing_obj, insight)
                    updated += 1
                else:
                    obj = insights_svc._parse_insight_to_history(config.id, insight)
                    db.add(obj)
                    inserted += 1

                if (inserted + updated) % BATCH_SIZE == 0:
                    await db.flush()

            except Exception as e:
                logger.error("Erro ao processar insight", ad_id=insight.get("ad_id"), error=str(e))
                errors += 1

        await db.flush()

        return {
            "synced": len(fb_insights),
            "inserted": inserted,
            "updated": updated,
            "errors": errors,
        }

    finally:
        await graph_client.close()


async def sync_all_insights(db: AsyncSession, config: SistemaFacebookAdsConfig) -> dict:
    """Sincroniza todos os insights historicos em chunks mensais."""
    chunks = generate_monthly_chunks(DATE_START, DATE_END, CHUNK_DAYS)
    total_chunks = len(chunks)

    totals = {"synced": 0, "inserted": 0, "updated": 0, "errors": 0}

    print(f"\n  Insights: {total_chunks} chunks de ~{CHUNK_DAYS} dias")
    print(f"  Periodo: {DATE_START} ate {DATE_END}\n")

    for i, (since, until) in enumerate(chunks, 1):
        print(f"  [{i}/{total_chunks}] {since} -> {until} ... ", end="", flush=True)

        try:
            result = await sync_insights_chunk(db, config, since, until)
            totals["synced"] += result["synced"]
            totals["inserted"] += result["inserted"]
            totals["updated"] += result["updated"]
            totals["errors"] += result["errors"]

            print(f"OK ({result['synced']} registros: "
                  f"{result['inserted']} novos, {result['updated']} atualizados"
                  f"{', ' + str(result['errors']) + ' erros' if result['errors'] else ''})")

        except Exception as e:
            totals["errors"] += 1
            print(f"ERRO: {e}")
            logger.error("Erro no chunk de insights", since=since, until=until, error=str(e))
            # Continua com o proximo chunk
            continue

    return totals


async def sync_today_insights(db: AsyncSession, config: SistemaFacebookAdsConfig) -> dict:
    """Sincroniza insights do dia atual na tabela today."""
    insights_svc = SyncInsightsService(db)
    print(f"\n  Sincronizando insights de hoje...")
    result = await insights_svc.sync_today(config)
    print(f"  Hoje: {result.get('synced', 0)} registros inseridos")
    return result


async def count_records(db: AsyncSession, config_id: int) -> dict:
    """Conta registros em cada tabela para verificacao."""
    counts = {}

    for model, name in [
        (SistemaFacebookAdsCampaigns, "campaigns"),
        (SistemaFacebookAdsAdsets, "adsets"),
        (SistemaFacebookAdsAds, "ads"),
        (SistemaFacebookAdsInsightsHistory, "insights_history"),
        (SistemaFacebookAdsInsightsToday, "insights_today"),
    ]:
        result = await db.execute(
            select(func.count(model.id)).where(model.config_id == config_id)
        )
        counts[name] = result.scalar_one()

    # Date range dos insights
    result = await db.execute(
        select(
            func.min(SistemaFacebookAdsInsightsHistory.date),
            func.max(SistemaFacebookAdsInsightsHistory.date),
        ).where(SistemaFacebookAdsInsightsHistory.config_id == config_id)
    )
    row = result.one_or_none()
    if row and row[0]:
        counts["insights_date_min"] = str(row[0])
        counts["insights_date_max"] = str(row[1])

    return counts


async def create_sync_history(db: AsyncSession, config_id: int, results: dict) -> None:
    """Cria registro de historico de sync."""
    sync_history = SistemaFacebookAdsSyncHistory(
        config_id=config_id,
        sync_type="historical",
        status=SyncStatus.COMPLETED.value,
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        campaigns_synced=results.get("campaigns", {}).get("synced", 0),
        adsets_synced=results.get("adsets", {}).get("synced", 0),
        ads_synced=results.get("ads", {}).get("synced", 0),
        insights_synced=results.get("insights", {}).get("synced", 0),
        entities_synced=(
            results.get("campaigns", {}).get("synced", 0) +
            results.get("adsets", {}).get("synced", 0) +
            results.get("ads", {}).get("synced", 0) +
            results.get("insights", {}).get("synced", 0)
        ),
        date_range_start=datetime.combine(DATE_START, datetime.min.time()),
        date_range_end=datetime.combine(DATE_END, datetime.min.time()),
    )
    db.add(sync_history)
    await db.flush()


async def main(config_id: int = None, dry_run: bool = False):
    """Executa backfill completo."""
    print("=" * 70)
    print("  FACEBOOK ADS - BACKFILL COMPLETO")
    print(f"  Periodo: {DATE_START} ate {DATE_END}")
    print(f"  Chunks de insights: {CHUNK_DAYS} dias")
    print("=" * 70)

    async with isolated_async_session() as db:
        # 1. Buscar configs
        configs = await get_active_configs(db, config_id)

        if not configs:
            print("\nNenhuma configuracao ativa encontrada!")
            if config_id:
                print(f"Config ID {config_id} nao encontrada ou inativa.")
            return

        print(f"\nConfiguracoes encontradas: {len(configs)}")
        for cfg in configs:
            print(f"  - ID={cfg.id} | Conta={cfg.account_id} | Nome={cfg.account_name}")

        if dry_run:
            print("\n[DRY RUN] Nenhuma alteracao sera feita.")
            return

        # 2. Processar cada config
        for config in configs:
            print(f"\n{'=' * 70}")
            print(f"  Config ID={config.id} | Conta: {config.account_name} ({config.account_id})")
            print(f"{'=' * 70}")

            all_results = {}
            start_time = datetime.utcnow()

            try:
                # Normalizar account_id (remover prefixo act_ se presente,
                # pois os clients do Facebook ja adicionam act_)
                original_account_id = config.account_id
                config.account_id = strip_act_prefix(config.account_id)
                print(f"  Account ID normalizado: {config.account_id}")

                # 2a. Campanhas, Ad Sets, Ads
                print(f"\n--- FASE 1: Entidades (Campanhas > Ad Sets > Ads) ---")
                entity_results = await sync_campaigns_adsets_ads(db, config)
                all_results.update(entity_results)

                # 2b. Insights historicos
                print(f"\n--- FASE 2: Insights Historicos ({DATE_START} a {DATE_END}) ---")
                insights_results = await sync_all_insights(db, config)
                all_results["insights"] = insights_results

                # 2c. Insights de hoje
                print(f"\n--- FASE 3: Insights de Hoje ---")
                today_results = await sync_today_insights(db, config)

                # 2d. Criar registro de sync
                await create_sync_history(db, config.id, all_results)

                # Restaurar account_id original antes do commit
                config.account_id = original_account_id

                # 2e. Commit
                await db.commit()

                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()

                # 3. Verificacao final
                print(f"\n--- VERIFICACAO FINAL ---")
                counts = await count_records(db, config.id)
                print(f"\n  Tabela                    | Registros")
                print(f"  --------------------------|----------")
                print(f"  Campanhas                 | {counts['campaigns']:>8}")
                print(f"  Conjuntos de Anuncios     | {counts['adsets']:>8}")
                print(f"  Anuncios                  | {counts['ads']:>8}")
                print(f"  Insights (historico)       | {counts['insights_history']:>8}")
                print(f"  Insights (hoje)            | {counts['insights_today']:>8}")
                if "insights_date_min" in counts:
                    print(f"  Periodo insights          | {counts['insights_date_min']} a {counts['insights_date_max']}")
                print(f"\n  Duracao total: {duration:.1f}s")

            except Exception as e:
                print(f"\n  ERRO FATAL: {e}")
                logger.error("Erro no backfill", config_id=config.id, error=str(e))
                await db.rollback()
                raise

    print(f"\n{'=' * 70}")
    print("  BACKFILL CONCLUIDO COM SUCESSO!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facebook Ads - Backfill Completo")
    parser.add_argument("--config-id", type=int, help="ID da configuracao (default: todas ativas)")
    parser.add_argument("--dry-run", action="store_true", help="Apenas mostra configs, nao executa")
    args = parser.parse_args()

    asyncio.run(main(config_id=args.config_id, dry_run=args.dry_run))
