"""
Serviço de dados para ETL e preparação.
Carrega dados do Facebook Ads e prepara para os modelos de ML.
"""

from datetime import datetime, timedelta
from typing import Optional
from decimal import Decimal

import pandas as pd
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.famachat_readonly import (
    SistemaFacebookAdsConfig,
    SistemaFacebookAdsCampaigns,
    SistemaFacebookAdsAdsets,
    SistemaFacebookAdsInsightsHistory,
    SistemaFacebookAdsInsightsToday,
)
from app.services.feature_engineering import (
    FeatureEngineer,
    CampaignFeatures,
    feature_engineer,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


class DataService:
    """
    Serviço para carregar e preparar dados do Facebook Ads.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.feature_engineer = feature_engineer
    
    async def get_campaign_daily_data(
        self,
        config_id: int,
        campaign_id: str,
        days: int = 30,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Obtém dados diários de uma campanha específica.
        
        Args:
            config_id: ID da configuração FB Ads
            campaign_id: ID da campanha no Facebook
            days: Número de dias de histórico
            end_date: Data final (default: hoje)
            
        Returns:
            DataFrame com dados diários
        """
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=days)
        
        # Query nos insights históricos
        query = select(
            SistemaFacebookAdsInsightsHistory.date,
            SistemaFacebookAdsInsightsHistory.spend,
            SistemaFacebookAdsInsightsHistory.impressions,
            SistemaFacebookAdsInsightsHistory.clicks,
            SistemaFacebookAdsInsightsHistory.leads,
            SistemaFacebookAdsInsightsHistory.frequency,
            SistemaFacebookAdsInsightsHistory.reach,
        ).where(
            and_(
                SistemaFacebookAdsInsightsHistory.config_id == config_id,
                SistemaFacebookAdsInsightsHistory.campaign_id == campaign_id,
                SistemaFacebookAdsInsightsHistory.date >= start_date,
                SistemaFacebookAdsInsightsHistory.date <= end_date,
            )
        ).order_by(SistemaFacebookAdsInsightsHistory.date.desc())
        
        result = await self.session.execute(query)
        rows = result.fetchall()
        
        if not rows:
            logger.warning(
                "Sem dados históricos para campanha",
                config_id=config_id,
                campaign_id=campaign_id
            )
            return pd.DataFrame()
        
        # Converter para DataFrame
        df = pd.DataFrame(rows, columns=[
            'date', 'spend', 'impressions', 'clicks', 'leads', 'frequency', 'reach'
        ])
        
        # Converter tipos
        df['date'] = pd.to_datetime(df['date'])
        df['spend'] = df['spend'].astype(float)
        df['impressions'] = df['impressions'].fillna(0).astype(int)
        df['clicks'] = df['clicks'].fillna(0).astype(int)
        df['leads'] = df['leads'].fillna(0).astype(int)
        df['frequency'] = df['frequency'].fillna(0).astype(float)
        df['reach'] = df['reach'].fillna(0).astype(int)
        
        return df
    
    async def get_all_campaigns_data(
        self,
        config_id: int,
        days: int = 30,
        active_only: bool = True
    ) -> pd.DataFrame:
        """
        Obtém dados de todas as campanhas de uma configuração.
        
        Args:
            config_id: ID da configuração FB Ads
            days: Número de dias de histórico
            active_only: Filtrar apenas campanhas ativas
            
        Returns:
            DataFrame com dados de todas as campanhas
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Buscar campanhas
        campaign_query = select(SistemaFacebookAdsCampaigns).where(
            SistemaFacebookAdsCampaigns.config_id == config_id
        )
        if active_only:
            campaign_query = campaign_query.where(
                SistemaFacebookAdsCampaigns.status == 'ACTIVE'
            )
        
        campaign_result = await self.session.execute(campaign_query)
        campaigns = list(campaign_result.scalars().all())
        
        if not campaigns:
            logger.info("Nenhuma campanha encontrada", config_id=config_id)
            return pd.DataFrame()
        
        campaign_ids = [c.campaign_id for c in campaigns]
        
        # Query nos insights
        query = select(
            SistemaFacebookAdsInsightsHistory.campaign_id,
            SistemaFacebookAdsInsightsHistory.date,
            SistemaFacebookAdsInsightsHistory.spend,
            SistemaFacebookAdsInsightsHistory.impressions,
            SistemaFacebookAdsInsightsHistory.clicks,
            SistemaFacebookAdsInsightsHistory.leads,
            SistemaFacebookAdsInsightsHistory.frequency,
            SistemaFacebookAdsInsightsHistory.reach,
        ).where(
            and_(
                SistemaFacebookAdsInsightsHistory.config_id == config_id,
                SistemaFacebookAdsInsightsHistory.campaign_id.in_(campaign_ids),
                SistemaFacebookAdsInsightsHistory.date >= start_date,
            )
        )
        
        result = await self.session.execute(query)
        rows = result.fetchall()
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=[
            'campaign_id', 'date', 'spend', 'impressions', 'clicks', 
            'leads', 'frequency', 'reach'
        ])
        
        # Converter tipos
        df['date'] = pd.to_datetime(df['date'])
        df['spend'] = df['spend'].astype(float)
        df['impressions'] = df['impressions'].fillna(0).astype(int)
        df['clicks'] = df['clicks'].fillna(0).astype(int)
        df['leads'] = df['leads'].fillna(0).astype(int)
        
        return df
    
    async def get_campaign_features(
        self,
        config_id: int,
        campaign_id: str,
        days: int = 30
    ) -> Optional[CampaignFeatures]:
        """
        Obtém features calculadas para uma campanha.
        
        Args:
            config_id: ID da configuração
            campaign_id: ID da campanha
            days: Dias de histórico para usar
            
        Returns:
            CampaignFeatures ou None se não houver dados
        """
        # Obter dados diários
        daily_data = await self.get_campaign_daily_data(
            config_id, campaign_id, days
        )
        
        if daily_data.empty:
            return None
        
        # Obter info da campanha
        campaign = await self._get_campaign_info(config_id, campaign_id)
        if not campaign:
            return None
        
        campaign_info = {
            'campaign_id': campaign_id,
            'config_id': config_id,
            'status': campaign.status,
            'daily_budget': campaign.daily_budget,
            'lifetime_budget': campaign.lifetime_budget,
        }
        
        # Calcular features
        return self.feature_engineer.compute_campaign_features(
            daily_data, campaign_info
        )
    
    async def get_all_campaign_features(
        self,
        config_id: int,
        active_only: bool = True
    ) -> list[CampaignFeatures]:
        """
        Obtém features de todas as campanhas de uma configuração.
        
        Args:
            config_id: ID da configuração
            active_only: Filtrar apenas campanhas ativas
            
        Returns:
            Lista de CampaignFeatures
        """
        # Buscar campanhas
        campaign_query = select(SistemaFacebookAdsCampaigns).where(
            SistemaFacebookAdsCampaigns.config_id == config_id
        )
        if active_only:
            campaign_query = campaign_query.where(
                SistemaFacebookAdsCampaigns.status == 'ACTIVE'
            )
        
        result = await self.session.execute(campaign_query)
        campaigns = list(result.scalars().all())
        
        features_list = []
        for campaign in campaigns:
            features = await self.get_campaign_features(
                config_id, campaign.campaign_id
            )
            if features:
                features_list.append(features)
        
        logger.info(
            "Features calculadas para campanhas",
            config_id=config_id,
            total_campaigns=len(campaigns),
            campaigns_with_features=len(features_list)
        )
        
        return features_list
    
    async def get_aggregated_metrics(
        self,
        config_id: int,
        days: int = 7
    ) -> dict:
        """
        Obtém métricas agregadas de referência para uma configuração.
        Usado para comparar campanhas individuais com a média.
        
        Args:
            config_id: ID da configuração
            days: Período para agregação
            
        Returns:
            Dict com métricas agregadas
        """
        data = await self.get_all_campaigns_data(config_id, days, active_only=False)
        
        if data.empty:
            return {
                'avg_cpl': 0.0,
                'avg_ctr': 0.0,
                'avg_cpc': 0.0,
                'total_spend': 0.0,
                'total_leads': 0,
                'campaigns_count': 0
            }
        
        return self.feature_engineer.compute_aggregated_metrics(data, days)
    
    async def get_config_summary(self, config_id: int) -> Optional[dict]:
        """
        Obtém resumo de uma configuração FB Ads.
        
        Returns:
            Dict com informações da config ou None
        """
        query = select(SistemaFacebookAdsConfig).where(
            SistemaFacebookAdsConfig.id == config_id
        )
        result = await self.session.execute(query)
        config = result.scalar_one_or_none()
        
        if not config:
            return None
        
        # Contar campanhas
        campaigns_query = select(func.count(SistemaFacebookAdsCampaigns.id)).where(
            SistemaFacebookAdsCampaigns.config_id == config_id
        )
        campaigns_result = await self.session.execute(campaigns_query)
        total_campaigns = campaigns_result.scalar() or 0
        
        # Contar campanhas ativas
        active_query = select(func.count(SistemaFacebookAdsCampaigns.id)).where(
            and_(
                SistemaFacebookAdsCampaigns.config_id == config_id,
                SistemaFacebookAdsCampaigns.status == 'ACTIVE'
            )
        )
        active_result = await self.session.execute(active_query)
        active_campaigns = active_result.scalar() or 0
        
        return {
            'id': config.id,
            'name': config.name,
            'ad_account_id': config.ad_account_id,
            'is_active': config.is_active,
            'total_campaigns': total_campaigns,
            'active_campaigns': active_campaigns,
        }
    
    async def _get_campaign_info(
        self,
        config_id: int,
        campaign_id: str
    ) -> Optional[SistemaFacebookAdsCampaigns]:
        """Obtém informações de uma campanha."""
        query = select(SistemaFacebookAdsCampaigns).where(
            and_(
                SistemaFacebookAdsCampaigns.config_id == config_id,
                SistemaFacebookAdsCampaigns.campaign_id == campaign_id
            )
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def check_data_availability(
        self,
        config_id: int,
        min_days: int = 7
    ) -> dict:
        """
        Verifica disponibilidade de dados para ML.
        
        Args:
            config_id: ID da configuração
            min_days: Mínimo de dias necessários
            
        Returns:
            Dict com informações de disponibilidade
        """
        cutoff = datetime.now() - timedelta(days=min_days)
        
        # Verificar dias com dados (contar apenas datas únicas, ignorando hora)
        from sqlalchemy import cast, Date
        query = select(
            func.count(func.distinct(cast(SistemaFacebookAdsInsightsHistory.date, Date)))
        ).where(
            and_(
                SistemaFacebookAdsInsightsHistory.config_id == config_id,
                SistemaFacebookAdsInsightsHistory.date >= cutoff,
            )
        )
        result = await self.session.execute(query)
        days_with_data = result.scalar() or 0
        
        # Verificar último sync
        last_sync_query = select(
            func.max(SistemaFacebookAdsInsightsHistory.consolidated_at)
        ).where(
            SistemaFacebookAdsInsightsHistory.config_id == config_id
        )
        last_sync_result = await self.session.execute(last_sync_query)
        last_sync = last_sync_result.scalar()
        
        has_enough_data = days_with_data >= min_days
        is_data_fresh = last_sync and (datetime.now() - last_sync).days < 2
        
        return {
            'config_id': config_id,
            'days_with_data': days_with_data,
            'min_days_required': min_days,
            'has_enough_data': has_enough_data,
            'last_sync': last_sync.isoformat() if last_sync else None,
            'is_data_fresh': is_data_fresh,
            'ready_for_ml': has_enough_data and is_data_fresh
        }


async def get_data_service(session: AsyncSession) -> DataService:
    """Factory para criar DataService."""
    return DataService(session)
