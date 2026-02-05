#!/usr/bin/env python3
"""
Script para sincronizar insights do Facebook Ads de julho de 2025.
Período: 01/07/2025 a 31/07/2025

Busca dados de campanhas, conjuntos de anúncios e anúncios a nível de anúncio
e salva na tabela sistema_facebook_ads_insights_history.

Uso:
    python scripts/sync_july_2025.py
    python scripts/sync_july_2025.py --config-id 1
    python scripts/sync_july_2025.py --dry-run
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime, date
from decimal import Decimal

# Garantir que o path do projeto está no sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from shared.core.logging import get_logger
from shared.db.session import isolated_async_session
from shared.db.models.famachat_readonly import (
    SistemaFacebookAdsConfig,
    SistemaFacebookAdsInsightsHistory,
)
from projects.facebook_ads.services.sync_insights import SyncInsightsService
from projects.facebook_ads.client.base import FacebookGraphClient
from projects.facebook_ads.client.insights import InsightsClient
from projects.facebook_ads.security.token_encryption import decrypt_token

logger = get_logger(__name__)

# === Configuração do período ===
DATE_START = date(2025, 7, 1)
DATE_END = date(2025, 7, 31)
BATCH_SIZE = 500


async def get_active_configs(db: AsyncSession, config_id: int = None) -> list:
    """Busca configurações ativas do Facebook Ads."""
    query = select(SistemaFacebookAdsConfig).where(
        SistemaFacebookAdsConfig.is_active == True
    )
    if config_id:
        query = query.where(SistemaFacebookAdsConfig.id == config_id)

    result = await db.execute(query)
    configs = result.scalars().all()
    return configs


async def sync_july_insights(
    db: AsyncSession,
    config: SistemaFacebookAdsConfig,
    dry_run: bool = False,
) -> dict:
    """Sincroniza insights de julho de 2025 para a tabela history."""
    access_token = decrypt_token(config.access_token)
    graph_client = FacebookGraphClient(access_token, config.account_id)
    insights_client = InsightsClient(graph_client)
    insights_svc = SyncInsightsService(db)

    time_range = {
        "since": DATE_START.strftime("%Y-%m-%d"),
        "until": DATE_END.strftime("%Y-%m-%d"),
    }

    print(f"\n  Buscando insights do período {DATE_START} a {DATE_END}...")

    try:
        # Usar relatório assíncrono para período de 31 dias
        account_id = config.account_id if config.account_id.startswith("act_") else f"act_{config.account_id}"
        
        print(f"  Conta: {account_id}")
        print(f"  Iniciando busca de dados na API do Facebook...")
        
        fb_insights = await insights_client.get_insights_async(
            account_id,
            time_range=time_range,
            level="ad",
        )

        print(f"  Encontrados {len(fb_insights)} registros de insights")

        if dry_run:
            print("\n  [DRY-RUN] Nenhum dado será salvo no banco")
            # Mostrar resumo dos dados
            campaigns = set()
            adsets = set()
            ads = set()
            total_spend = Decimal("0")
            total_impressions = 0
            total_clicks = 0
            
            for insight in fb_insights:
                campaigns.add(insight.get("campaign_id", ""))
                adsets.add(insight.get("adset_id", ""))
                ads.add(insight.get("ad_id", ""))
                total_spend += Decimal(str(insight.get("spend", "0")))
                total_impressions += int(insight.get("impressions", 0))
                total_clicks += int(insight.get("clicks", 0))
            
            print(f"\n  === Resumo dos Dados ===")
            print(f"  Campanhas únicas: {len(campaigns)}")
            print(f"  Conjuntos de anúncios únicos: {len(adsets)}")
            print(f"  Anúncios únicos: {len(ads)}")
            print(f"  Total de registros (dias x anúncios): {len(fb_insights)}")
            print(f"  Gasto total: R$ {total_spend:,.2f}")
            print(f"  Impressões totais: {total_impressions:,}")
            print(f"  Cliques totais: {total_clicks:,}")
            
            return {
                "synced": len(fb_insights),
                "inserted": 0,
                "updated": 0,
                "errors": 0,
                "dry_run": True,
            }

        inserted = 0
        updated = 0
        errors = 0

        print(f"\n  Processando e salvando no banco de dados...")

        for i, insight in enumerate(fb_insights, 1):
            try:
                insight_date = insight.get("date_start", "")
                ad_id = insight.get("ad_id", "")

                if not insight_date or not ad_id:
                    continue

                # Verificar se já existe
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
                    print(f"    Processados {i}/{len(fb_insights)} registros...")

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


async def count_july_records(db: AsyncSession, config_id: int) -> dict:
    """Conta registros de julho 2025 na tabela de insights."""
    result = await db.execute(
        select(SistemaFacebookAdsInsightsHistory).where(
            and_(
                SistemaFacebookAdsInsightsHistory.config_id == config_id,
                SistemaFacebookAdsInsightsHistory.date >= datetime(2025, 7, 1),
                SistemaFacebookAdsInsightsHistory.date <= datetime(2025, 7, 31),
            )
        )
    )
    records = result.scalars().all()
    return {
        "total": len(records),
        "unique_ads": len(set(r.ad_id for r in records)),
        "unique_adsets": len(set(r.adset_id for r in records)),
        "unique_campaigns": len(set(r.campaign_id for r in records)),
    }


async def main(config_id: int = None, dry_run: bool = False):
    """Executa a sincronização de julho 2025."""
    print("=" * 70)
    print("  SINCRONIZAÇÃO DE INSIGHTS - JULHO 2025")
    print(f"  Período: {DATE_START} a {DATE_END}")
    print("=" * 70)

    async with isolated_async_session() as db:
        # Buscar configurações
        configs = await get_active_configs(db, config_id)

        if not configs:
            print("\n❌ Nenhuma conta Facebook Ads ativa encontrada.")
            return

        print(f"\n✓ {len(configs)} conta(s) ativa(s) encontrada(s)")

        for config in configs:
            print(f"\n{'=' * 70}")
            print(f"  Conta: {config.account_name} (ID: {config.id})")
            print(f"  Account ID: {config.account_id}")
            print("=" * 70)

            # Contagem antes
            if not dry_run:
                before_count = await count_july_records(db, config.id)
                print(f"\n  Registros existentes para julho/2025: {before_count['total']}")

            try:
                result = await sync_july_insights(db, config, dry_run=dry_run)

                print(f"\n  === Resultado ===")
                print(f"  Registros sincronizados: {result['synced']}")
                print(f"  Novos registros inseridos: {result['inserted']}")
                print(f"  Registros atualizados: {result['updated']}")
                if result['errors'] > 0:
                    print(f"  ⚠️  Erros: {result['errors']}")

                # Contagem depois
                if not dry_run:
                    after_count = await count_july_records(db, config.id)
                    print(f"\n  Total de registros após sync: {after_count['total']}")
                    print(f"  - Campanhas únicas: {after_count['unique_campaigns']}")
                    print(f"  - Conjuntos de anúncios únicos: {after_count['unique_adsets']}")
                    print(f"  - Anúncios únicos: {after_count['unique_ads']}")

            except Exception as e:
                print(f"\n❌ Erro ao sincronizar: {e}")
                logger.exception("Erro na sincronização", config_id=config.id)

    print(f"\n{'=' * 70}")
    print("  SINCRONIZAÇÃO CONCLUÍDA!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sincroniza insights do Facebook Ads de julho 2025"
    )
    parser.add_argument(
        "--config-id",
        type=int,
        help="ID específico da conta a sincronizar",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas mostra o que seria feito, sem salvar no banco",
    )

    args = parser.parse_args()

    try:
        asyncio.run(main(config_id=args.config_id, dry_run=args.dry_run))
    except KeyboardInterrupt:
        print("\n\n⚠️  Sincronização interrompida pelo usuário.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        logger.exception("Erro fatal no script")
        sys.exit(1)
