"""
Repositório para acesso aos insights do Facebook Ads.
Acesso READ-ONLY às tabelas do FamaChat.
"""

from datetime import datetime, timedelta
from typing import Optional
from decimal import Decimal

import pandas as pd
from sqlalchemy import select, func, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.famachat_readonly import (
    SistemaFacebookAdsConfig,
    SistemaFacebookAdsCampaigns,
    SistemaFacebookAdsAdsets,
    SistemaFacebookAdsAds,
    SistemaFacebookAdsInsightsHistory,
    SistemaFacebookAdsInsightsToday,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


class InsightsRepository:
    """
    Repositório para acesso aos dados de Facebook Ads.
    Todas as operações são READ-ONLY.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    # ==================== CONFIGS ====================

    async def get_config(self, config_id: int) -> Optional[SistemaFacebookAdsConfig]:
        """Obtém configuração de FB Ads por ID."""
        result = await self.session.execute(
            select(SistemaFacebookAdsConfig)
            .where(SistemaFacebookAdsConfig.id == config_id)
        )
        return result.scalar_one_or_none()

    async def get_active_configs(self) -> list[SistemaFacebookAdsConfig]:
        """Obtém todas as configurações ativas."""
        result = await self.session.execute(
            select(SistemaFacebookAdsConfig)
            .where(SistemaFacebookAdsConfig.is_active == True)
        )
        return list(result.scalars().all())

    # ==================== CAMPAIGNS ====================

    async def get_campaign(
        self, config_id: int, campaign_id: str
    ) -> Optional[SistemaFacebookAdsCampaigns]:
        """Obtém uma campanha específica."""
        result = await self.session.execute(
            select(SistemaFacebookAdsCampaigns)
            .where(
                and_(
                    SistemaFacebookAdsCampaigns.config_id == config_id,
                    SistemaFacebookAdsCampaigns.campaign_id == campaign_id
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_active_campaigns(
        self, config_id: int
    ) -> list[SistemaFacebookAdsCampaigns]:
        """Obtém todas as campanhas ativas de uma configuração."""
        result = await self.session.execute(
            select(SistemaFacebookAdsCampaigns)
            .where(
                and_(
                    SistemaFacebookAdsCampaigns.config_id == config_id,
                    SistemaFacebookAdsCampaigns.status == "ACTIVE"
                )
            )
        )
        return list(result.scalars().all())

    async def get_all_campaigns(
        self, config_id: int
    ) -> list[SistemaFacebookAdsCampaigns]:
        """Obtém todas as campanhas de uma configuração."""
        result = await self.session.execute(
            select(SistemaFacebookAdsCampaigns)
            .where(SistemaFacebookAdsCampaigns.config_id == config_id)
        )
        return list(result.scalars().all())

    # ==================== INSIGHTS ====================

    async def get_insights_history(
        self,
        config_id: int,
        start_date: datetime,
        end_date: datetime,
        entity_type: str = "campaign",
        entity_id: Optional[str] = None,
    ) -> list[SistemaFacebookAdsInsightsHistory]:
        """
        Obtém insights históricos agregados.

        Args:
            config_id: ID da configuração
            start_date: Data inicial
            end_date: Data final
            entity_type: Tipo de entidade (campaign, adset, ad)
            entity_id: ID específico (opcional)
        """
        query = select(SistemaFacebookAdsInsightsHistory).where(
            and_(
                SistemaFacebookAdsInsightsHistory.config_id == config_id,
                SistemaFacebookAdsInsightsHistory.date >= start_date,
                SistemaFacebookAdsInsightsHistory.date <= end_date
            )
        )

        if entity_id:
            if entity_type == "campaign":
                query = query.where(
                    SistemaFacebookAdsInsightsHistory.campaign_id == entity_id
                )
            elif entity_type == "adset":
                query = query.where(
                    SistemaFacebookAdsInsightsHistory.adset_id == entity_id
                )
            elif entity_type == "ad":
                query = query.where(
                    SistemaFacebookAdsInsightsHistory.ad_id == entity_id
                )

        query = query.order_by(SistemaFacebookAdsInsightsHistory.date)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_insights_as_dataframe(
        self,
        config_id: int,
        start_date: datetime,
        end_date: datetime,
        entity_type: str = "campaign",
        entity_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Obtém insights históricos como DataFrame pandas.
        Agregado por data e entidade.
        """
        insights = await self.get_insights_history(
            config_id, start_date, end_date, entity_type, entity_id
        )

        if not insights:
            return pd.DataFrame()

        # Converter para DataFrame
        data = []
        for insight in insights:
            data.append({
                "date": insight.date,
                "campaign_id": insight.campaign_id,
                "adset_id": insight.adset_id,
                "ad_id": insight.ad_id,
                "impressions": insight.impressions or 0,
                "reach": insight.reach or 0,
                "frequency": float(insight.frequency or 0),
                "clicks": insight.clicks or 0,
                "spend": float(insight.spend or 0),
                "cpc": float(insight.cpc or 0),
                "cpm": float(insight.cpm or 0),
                "ctr": float(insight.ctr or 0),
                "leads": insight.leads or 0,
                "cost_per_lead": float(insight.cost_per_lead or 0),
                "conversions": insight.conversions or 0,
                "video_views": insight.video_views or 0,
                "post_engagement": insight.post_engagement or 0,
            })

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df

    async def get_aggregated_metrics_by_campaign(
        self,
        config_id: int,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Obtém métricas agregadas por campanha no período.
        Útil para classificação de campanhas.
        """
        df = await self.get_insights_as_dataframe(
            config_id, start_date, end_date, "campaign"
        )

        if df.empty:
            return df

        # Agregar por campanha
        agg_df = df.groupby("campaign_id").agg({
            "impressions": "sum",
            "reach": "sum",
            "clicks": "sum",
            "spend": "sum",
            "leads": "sum",
            "conversions": "sum",
            "video_views": "sum",
            "post_engagement": "sum",
        }).reset_index()

        # Calcular métricas derivadas
        agg_df["ctr"] = (agg_df["clicks"] / agg_df["impressions"] * 100).fillna(0)
        agg_df["cpc"] = (agg_df["spend"] / agg_df["clicks"]).fillna(0)
        agg_df["cpm"] = (agg_df["spend"] / agg_df["impressions"] * 1000).fillna(0)
        agg_df["cpl"] = (agg_df["spend"] / agg_df["leads"]).replace([float('inf')], 0).fillna(0)
        agg_df["days_active"] = df.groupby("campaign_id")["date"].nunique().values

        return agg_df

    async def get_daily_totals(
        self,
        config_id: int,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Obtém totais diários para time series.
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        df = await self.get_insights_as_dataframe(
            config_id, start_date, end_date
        )

        if df.empty:
            return df

        # Agregar por dia
        daily = df.groupby("date").agg({
            "spend": "sum",
            "leads": "sum",
            "impressions": "sum",
            "clicks": "sum",
        }).reset_index()

        daily["cpl"] = (daily["spend"] / daily["leads"]).replace([float('inf')], 0).fillna(0)
        daily["ctr"] = (daily["clicks"] / daily["impressions"] * 100).fillna(0)

        return daily

    async def get_average_metrics(
        self,
        config_id: int,
        days: int = 30,
    ) -> dict:
        """
        Obtém métricas médias do período para comparação.
        """
        daily = await self.get_daily_totals(config_id, days)

        if daily.empty:
            return {
                "avg_cpl": 0,
                "avg_ctr": 0,
                "avg_daily_spend": 0,
                "avg_daily_leads": 0,
                "total_spend": 0,
                "total_leads": 0,
            }

        return {
            "avg_cpl": daily["cpl"].mean(),
            "avg_ctr": daily["ctr"].mean(),
            "avg_daily_spend": daily["spend"].mean(),
            "avg_daily_leads": daily["leads"].mean(),
            "total_spend": daily["spend"].sum(),
            "total_leads": daily["leads"].sum(),
        }

    async def count_insights_samples(
        self,
        config_id: int,
        entity_type: str = "campaign",
        entity_id: Optional[str] = None,
    ) -> int:
        """Conta número de amostras disponíveis para treinamento."""
        query = select(func.count(SistemaFacebookAdsInsightsHistory.id)).where(
            SistemaFacebookAdsInsightsHistory.config_id == config_id
        )

        if entity_id:
            if entity_type == "campaign":
                query = query.where(
                    SistemaFacebookAdsInsightsHistory.campaign_id == entity_id
                )

        result = await self.session.execute(query)
        return result.scalar() or 0
