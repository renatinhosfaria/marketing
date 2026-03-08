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

from shared.db.models.famachat_readonly import (
    SistemaFacebookAdsConfig,
    SistemaFacebookAdsCampaigns,
    SistemaFacebookAdsAdsets,
    SistemaFacebookAdsAds,
    SistemaFacebookAdsInsightsHistory,
    SistemaFacebookAdsInsightsToday,
)
from projects.ml.services.feature_engineering import (
    FeatureEngineer,
    CampaignFeatures,
    EntityFeatures,
    feature_engineer,
)
from shared.core.logging import get_logger

logger = get_logger(__name__)


class DataService:
    """
    Serviço para carregar e preparar dados do Facebook Ads.
    Inclui dados do dia atual (insights_today) para detecção em tempo quase real.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.feature_engineer = feature_engineer

    # =========================================================================
    # Helpers para integração de dados do dia atual (insights_today)
    # =========================================================================

    async def _fetch_today_data(
        self,
        config_id: int,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        campaign_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Busca dados do dia atual da tabela insights_today.
        Agrega por entidade/data para consolidar múltiplos ads.

        Args:
            config_id: ID da configuração
            entity_type: Tipo da entidade para filtro (campaign, adset, ad)
            entity_id: ID da entidade para filtro
            campaign_ids: Lista de campaign IDs para filtro (usado em get_all)

        Returns:
            DataFrame com mesmas colunas do history
        """
        T = SistemaFacebookAdsInsightsToday

        # Determinar a coluna de agrupamento
        group_col = T.campaign_id  # default

        filters = [T.config_id == config_id]

        if entity_type and entity_id:
            if entity_type == "campaign":
                filters.append(T.campaign_id == entity_id)
                group_col = T.campaign_id
            elif entity_type == "adset":
                filters.append(T.adset_id == entity_id)
                group_col = T.adset_id
            elif entity_type == "ad":
                filters.append(T.ad_id == entity_id)
                group_col = T.ad_id

        if campaign_ids:
            filters.append(T.campaign_id.in_(campaign_ids))

        query = select(
            func.date(T.date).label('date'),
            T.campaign_id,
            T.adset_id,
            T.ad_id,
            func.sum(T.spend).label('spend'),
            func.sum(T.impressions).label('impressions'),
            func.sum(T.clicks).label('clicks'),
            func.sum(T.leads).label('leads'),
            func.avg(T.frequency).label('frequency'),
            func.sum(T.reach).label('reach'),
        ).where(
            and_(*filters)
        ).group_by(
            func.date(T.date), T.campaign_id, T.adset_id, T.ad_id
        )

        result = await self.session.execute(query)
        rows = result.fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            'date', 'campaign_id', 'adset_id', 'ad_id',
            'spend', 'impressions', 'clicks', 'leads', 'frequency', 'reach'
        ])

        df['date'] = pd.to_datetime(df['date'])
        df['spend'] = df['spend'].astype(float)
        df['impressions'] = df['impressions'].fillna(0).astype(int)
        df['clicks'] = df['clicks'].fillna(0).astype(int)
        df['leads'] = df['leads'].fillna(0).astype(int)
        df['frequency'] = df['frequency'].fillna(0).astype(float)
        df['reach'] = df['reach'].fillna(0).astype(int)

        return df

    def _merge_today_with_history(
        self,
        history_df: pd.DataFrame,
        today_df: pd.DataFrame,
        group_cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Faz merge dos dados de today com history.
        Dados do history prevalecem para datas que já existem.

        Args:
            history_df: DataFrame com dados históricos
            today_df: DataFrame com dados de hoje
            group_cols: Colunas de agrupamento adicionais (ex: campaign_id)

        Returns:
            DataFrame combinado
        """
        if today_df.empty:
            return history_df
        if history_df.empty:
            return today_df

        # Normalizar datas para comparação (só a parte de data, sem hora)
        history_dates = set(history_df['date'].dt.date)

        # Filtrar today para manter apenas datas que NÃO existem no history
        today_df = today_df.copy()
        today_df['_date_only'] = today_df['date'].dt.date
        new_data = today_df[~today_df['_date_only'].isin(history_dates)].drop(
            columns=['_date_only']
        )

        if new_data.empty:
            return history_df

        # Garantir mesmas colunas
        common_cols = [c for c in history_df.columns if c in new_data.columns]
        new_data = new_data[common_cols]

        # Concatenar e ordenar
        merged = pd.concat([history_df, new_data], ignore_index=True)
        merged = merged.sort_values('date').reset_index(drop=True)

        logger.debug(
            "Dados de hoje integrados ao histórico",
            history_rows=len(history_df),
            today_new_rows=len(new_data),
            total_rows=len(merged),
        )

        return merged
    
    async def get_campaign_daily_data(
        self,
        config_id: int,
        campaign_id: str,
        days: int = 60,
        end_date: Optional[datetime] = None,
        include_today: bool = True,
    ) -> pd.DataFrame:
        """
        Obtém dados diários de uma campanha específica.
        
        Args:
            config_id: ID da configuração FB Ads
            campaign_id: ID da campanha no Facebook
            days: Número de dias de histórico
            end_date: Data final (default: hoje)
            include_today: Se True, inclui dados do dia atual (insights_today)
            
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
        
        if not rows and not include_today:
            logger.warning(
                "Sem dados históricos para campanha",
                config_id=config_id,
                campaign_id=campaign_id
            )
            return pd.DataFrame()
        
        # Converter para DataFrame
        if rows:
            df = pd.DataFrame(rows, columns=[
                'date', 'spend', 'impressions', 'clicks', 'leads', 'frequency', 'reach'
            ])
            df['date'] = pd.to_datetime(df['date'])
            df['spend'] = df['spend'].astype(float)
            df['impressions'] = df['impressions'].fillna(0).astype(int)
            df['clicks'] = df['clicks'].fillna(0).astype(int)
            df['leads'] = df['leads'].fillna(0).astype(int)
            df['frequency'] = df['frequency'].fillna(0).astype(float)
            df['reach'] = df['reach'].fillna(0).astype(int)
        else:
            df = pd.DataFrame()

        # Integrar dados do dia atual
        if include_today:
            today_df = await self._fetch_today_data(
                config_id=config_id,
                entity_type="campaign",
                entity_id=campaign_id,
            )
            if not today_df.empty:
                # Selecionar apenas colunas compatíveis
                today_cols = [c for c in ['date', 'spend', 'impressions', 'clicks',
                              'leads', 'frequency', 'reach'] if c in today_df.columns]
                today_subset = today_df[today_cols].copy()
                # Agregar por data (pode haver múltiplos ads por campanha)
                today_agg = today_subset.groupby('date').agg({
                    'spend': 'sum',
                    'impressions': 'sum',
                    'clicks': 'sum',
                    'leads': 'sum',
                    'frequency': 'mean',
                    'reach': 'sum',
                }).reset_index()
                df = self._merge_today_with_history(df, today_agg)

        if df.empty:
            logger.warning(
                "Sem dados para campanha",
                config_id=config_id,
                campaign_id=campaign_id
            )

        return df

    async def get_entity_daily_data(
        self,
        config_id: int,
        entity_type: str,
        entity_id: str,
        days: int = 60,
        end_date: Optional[datetime] = None,
        include_today: bool = True,
    ) -> pd.DataFrame:
        """
        Obtem dados diarios de uma entidade especifica.

        Args:
            config_id: ID da configuracao FB Ads
            entity_type: 'campaign', 'adset', ou 'ad'
            entity_id: ID da entidade no Facebook
            days: Numero de dias de historico
            end_date: Data final (default: hoje)
            include_today: Se True, inclui dados do dia atual (insights_today)

        Returns:
            DataFrame com dados diarios
        """
        if end_date is None:
            end_date = datetime.now()

        start_date = end_date - timedelta(days=days)

        # Build filter based on entity type
        entity_filter = None
        if entity_type == "campaign":
            entity_filter = SistemaFacebookAdsInsightsHistory.campaign_id == entity_id
        elif entity_type == "adset":
            entity_filter = SistemaFacebookAdsInsightsHistory.adset_id == entity_id
        elif entity_type == "ad":
            entity_filter = SistemaFacebookAdsInsightsHistory.ad_id == entity_id
        else:
            raise ValueError(f"Unknown entity_type: {entity_type}")

        query = select(
            SistemaFacebookAdsInsightsHistory.date,
            SistemaFacebookAdsInsightsHistory.campaign_id,
            SistemaFacebookAdsInsightsHistory.adset_id,
            SistemaFacebookAdsInsightsHistory.ad_id,
            SistemaFacebookAdsInsightsHistory.spend,
            SistemaFacebookAdsInsightsHistory.impressions,
            SistemaFacebookAdsInsightsHistory.clicks,
            SistemaFacebookAdsInsightsHistory.leads,
            SistemaFacebookAdsInsightsHistory.frequency,
            SistemaFacebookAdsInsightsHistory.reach,
        ).where(
            and_(
                SistemaFacebookAdsInsightsHistory.config_id == config_id,
                entity_filter,
                SistemaFacebookAdsInsightsHistory.date >= start_date,
                SistemaFacebookAdsInsightsHistory.date <= end_date,
            )
        ).order_by(SistemaFacebookAdsInsightsHistory.date.desc())

        result = await self.session.execute(query)
        rows = result.fetchall()

        if rows:
            df = pd.DataFrame(rows, columns=[
                'date', 'campaign_id', 'adset_id', 'ad_id',
                'spend', 'impressions', 'clicks', 'leads', 'frequency', 'reach'
            ])
            df['date'] = pd.to_datetime(df['date'])
            df['spend'] = df['spend'].astype(float)
            df['impressions'] = df['impressions'].fillna(0).astype(int)
            df['clicks'] = df['clicks'].fillna(0).astype(int)
            df['leads'] = df['leads'].fillna(0).astype(int)
            df['frequency'] = df['frequency'].fillna(0).astype(float)
            df['reach'] = df['reach'].fillna(0).astype(int)
        else:
            df = pd.DataFrame()

        # Integrar dados do dia atual
        if include_today:
            today_df = await self._fetch_today_data(
                config_id=config_id,
                entity_type=entity_type,
                entity_id=entity_id,
            )
            if not today_df.empty:
                df = self._merge_today_with_history(df, today_df)

        return df

    async def get_all_campaigns_data(
        self,
        config_id: int,
        days: int = 60,
        active_only: bool = True,
        include_today: bool = True,
    ) -> pd.DataFrame:
        """
        Obtém dados de todas as campanhas de uma configuração.
        
        Args:
            config_id: ID da configuração FB Ads
            days: Número de dias de histórico
            active_only: Filtrar apenas campanhas ativas
            include_today: Se True, inclui dados do dia atual (insights_today)
            
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
        
        # Query nos insights históricos
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
        
        if rows:
            df = pd.DataFrame(rows, columns=[
                'campaign_id', 'date', 'spend', 'impressions', 'clicks', 
                'leads', 'frequency', 'reach'
            ])
            df['date'] = pd.to_datetime(df['date'])
            df['spend'] = df['spend'].astype(float)
            df['impressions'] = df['impressions'].fillna(0).astype(int)
            df['clicks'] = df['clicks'].fillna(0).astype(int)
            df['leads'] = df['leads'].fillna(0).astype(int)
        else:
            df = pd.DataFrame()

        # Integrar dados do dia atual
        if include_today:
            today_df = await self._fetch_today_data(
                config_id=config_id,
                campaign_ids=campaign_ids,
            )
            if not today_df.empty:
                # Agregar por campaign_id + date
                today_agg = today_df.groupby(['campaign_id', 'date']).agg({
                    'spend': 'sum',
                    'impressions': 'sum',
                    'clicks': 'sum',
                    'leads': 'sum',
                    'frequency': 'mean',
                    'reach': 'sum',
                }).reset_index()
                df = self._merge_today_with_history(df, today_agg)

        return df
    
    async def get_campaign_features(
        self,
        config_id: int,
        campaign_id: str,
        days: int = 60
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

    async def get_entity_features(
        self,
        config_id: int,
        entity_type: str,
        entity_id: str,
        parent_id: Optional[str] = None,
        days: int = 60
    ) -> Optional[EntityFeatures]:
        """
        Obtem features calculadas para uma entidade.

        Args:
            config_id: ID da configuracao
            entity_type: 'campaign', 'adset', ou 'ad'
            entity_id: ID da entidade
            parent_id: ID do parent (para calcular share)
            days: Dias de historico

        Returns:
            EntityFeatures ou None se nao houver dados
        """
        daily_data = await self.get_entity_daily_data(
            config_id, entity_type, entity_id, days
        )

        if daily_data.empty:
            return None

        # Get entity info
        entity_info = await self._get_entity_info(config_id, entity_type, entity_id)
        if not entity_info:
            return None

        # Get parent metrics for hierarchical context
        parent_metrics = None
        if entity_type in ("adset", "ad") and parent_id:
            parent_type = "campaign" if entity_type == "adset" else "adset"
            parent_data = await self.get_entity_daily_data(
                config_id, parent_type, parent_id, days=7
            )
            if not parent_data.empty:
                parent_metrics = {
                    'spend': parent_data['spend'].sum(),
                    'leads': parent_data['leads'].sum(),
                }

        # Get sibling metrics for comparison
        sibling_metrics = None
        if entity_type in ("adset", "ad"):
            sibling_metrics = await self._get_sibling_metrics(
                config_id, entity_type, entity_id, parent_id, days=7
            )

        return self.feature_engineer.compute_entity_features(
            daily_data,
            entity_info,
            entity_type=entity_type,
            parent_metrics=parent_metrics,
            sibling_metrics=sibling_metrics,
        )

    async def get_all_entity_features(
        self,
        config_id: int,
        entity_type: str,
        parent_id: Optional[str] = None,
        active_only: bool = True
    ) -> list[EntityFeatures]:
        """
        Obtem features de todas as entidades de um tipo.

        Args:
            config_id: ID da configuracao
            entity_type: 'campaign', 'adset', ou 'ad'
            parent_id: Filtrar por parent (campaign_id para adsets, adset_id para ads)
            active_only: Filtrar apenas entidades ativas

        Returns:
            Lista de EntityFeatures
        """
        entities = await self._get_entities(config_id, entity_type, parent_id, active_only)

        features_list = []
        for entity in entities:
            entity_id = self._get_entity_id(entity, entity_type)
            parent = self._get_parent_id(entity, entity_type)

            features = await self.get_entity_features(
                config_id, entity_type, entity_id, parent_id=parent
            )
            if features:
                features_list.append(features)

        logger.info(
            "Features calculadas para entidades",
            config_id=config_id,
            entity_type=entity_type,
            total=len(entities),
            with_features=len(features_list)
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

    async def _get_entity_info(
        self,
        config_id: int,
        entity_type: str,
        entity_id: str
    ) -> Optional[dict]:
        """Obtem informacoes de uma entidade."""
        if entity_type == "campaign":
            campaign = await self._get_campaign_info(config_id, entity_id)
            if campaign:
                return {
                    'campaign_id': campaign.campaign_id,
                    'config_id': config_id,
                    'status': campaign.status,
                    'daily_budget': campaign.daily_budget,
                    'lifetime_budget': campaign.lifetime_budget,
                }
        elif entity_type == "adset":
            adset = await self._get_adset_info(config_id, entity_id)
            if adset:
                return {
                    'adset_id': adset.adset_id,
                    'campaign_id': adset.campaign_id,
                    'config_id': config_id,
                    'status': adset.status,
                    'daily_budget': adset.daily_budget,
                    'lifetime_budget': adset.lifetime_budget,
                }
        elif entity_type == "ad":
            ad = await self._get_ad_info(config_id, entity_id)
            if ad:
                return {
                    'ad_id': ad.ad_id,
                    'adset_id': ad.adset_id,
                    'config_id': config_id,
                    'status': ad.status,
                    'daily_budget': 0,
                    'lifetime_budget': 0,
                }
        return None

    async def _get_adset_info(
        self,
        config_id: int,
        adset_id: str
    ) -> Optional[SistemaFacebookAdsAdsets]:
        """Obtem informacoes de um adset."""
        query = select(SistemaFacebookAdsAdsets).where(
            and_(
                SistemaFacebookAdsAdsets.config_id == config_id,
                SistemaFacebookAdsAdsets.adset_id == adset_id
            )
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def _get_ad_info(
        self,
        config_id: int,
        ad_id: str
    ) -> Optional[SistemaFacebookAdsAds]:
        """Obtem informacoes de um ad."""
        query = select(SistemaFacebookAdsAds).where(
            and_(
                SistemaFacebookAdsAds.config_id == config_id,
                SistemaFacebookAdsAds.ad_id == ad_id
            )
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def _get_entities(
        self,
        config_id: int,
        entity_type: str,
        parent_id: Optional[str] = None,
        active_only: bool = True
    ) -> list:
        """Obtem lista de entidades por tipo."""
        if entity_type == "campaign":
            query = select(SistemaFacebookAdsCampaigns).where(
                SistemaFacebookAdsCampaigns.config_id == config_id
            )
            if active_only:
                query = query.where(SistemaFacebookAdsCampaigns.status == 'ACTIVE')
        elif entity_type == "adset":
            query = select(SistemaFacebookAdsAdsets).where(
                SistemaFacebookAdsAdsets.config_id == config_id
            )
            if parent_id:
                query = query.where(SistemaFacebookAdsAdsets.campaign_id == parent_id)
            if active_only:
                query = query.where(SistemaFacebookAdsAdsets.status == 'ACTIVE')
        elif entity_type == "ad":
            query = select(SistemaFacebookAdsAds).where(
                SistemaFacebookAdsAds.config_id == config_id
            )
            if parent_id:
                query = query.where(SistemaFacebookAdsAds.adset_id == parent_id)
            if active_only:
                query = query.where(SistemaFacebookAdsAds.status == 'ACTIVE')
        else:
            raise ValueError(f"Unknown entity_type: {entity_type}")

        result = await self.session.execute(query)
        return list(result.scalars().all())

    def _get_entity_id(self, entity, entity_type: str) -> str:
        """Extrai entity_id de uma entidade."""
        if entity_type == "campaign":
            return entity.campaign_id
        elif entity_type == "adset":
            return entity.adset_id
        elif entity_type == "ad":
            return entity.ad_id
        return str(entity.id)

    def _get_parent_id(self, entity, entity_type: str) -> Optional[str]:
        """Extrai parent_id de uma entidade."""
        if entity_type == "adset":
            return entity.campaign_id
        elif entity_type == "ad":
            return entity.adset_id
        return None

    async def get_entity_names(
        self,
        config_id: int,
        entity_type: str,
        entity_id: str
    ) -> dict:
        """
        Obtém os nomes da entidade e seus parents para identificação visual.

        Args:
            config_id: ID da configuração
            entity_type: 'campaign', 'adset', ou 'ad'
            entity_id: ID da entidade

        Returns:
            Dict com campaign_name, adset_name, ad_name conforme aplicável
        """
        result = {
            'campaign_name': None,
            'adset_name': None,
            'ad_name': None,
        }

        if entity_type == "campaign":
            campaign = await self._get_campaign_info(config_id, entity_id)
            if campaign:
                result['campaign_name'] = campaign.name
        elif entity_type == "adset":
            adset = await self._get_adset_info(config_id, entity_id)
            if adset:
                result['adset_name'] = adset.name
                # Get parent campaign name
                campaign = await self._get_campaign_info(config_id, adset.campaign_id)
                if campaign:
                    result['campaign_name'] = campaign.name
        elif entity_type == "ad":
            ad = await self._get_ad_info(config_id, entity_id)
            if ad:
                result['ad_name'] = ad.name
                # Get parent adset and campaign names
                adset = await self._get_adset_info(config_id, ad.adset_id)
                if adset:
                    result['adset_name'] = adset.name
                    campaign = await self._get_campaign_info(config_id, adset.campaign_id)
                    if campaign:
                        result['campaign_name'] = campaign.name

        return result

    async def _get_sibling_metrics(
        self,
        config_id: int,
        entity_type: str,
        entity_id: str,
        parent_id: Optional[str],
        days: int = 14
    ) -> Optional[pd.DataFrame]:
        """Obtem metricas dos irmaos para comparacao (inclui dados de hoje)."""
        if not parent_id:
            return None

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        T = SistemaFacebookAdsInsightsToday
        H = SistemaFacebookAdsInsightsHistory

        if entity_type == "adset":
            # History: all adsets in the same campaign
            query = select(
                H.adset_id.label('entity_id'),
                func.sum(H.spend).label('spend'),
                func.sum(H.leads).label('leads'),
            ).where(
                and_(
                    H.config_id == config_id,
                    H.campaign_id == parent_id,
                    H.adset_id != entity_id,
                    H.date >= start_date,
                )
            ).group_by(H.adset_id)

            # Today: same filter
            today_query = select(
                T.adset_id.label('entity_id'),
                func.sum(T.spend).label('spend'),
                func.sum(T.leads).label('leads'),
            ).where(
                and_(
                    T.config_id == config_id,
                    T.campaign_id == parent_id,
                    T.adset_id != entity_id,
                )
            ).group_by(T.adset_id)

        elif entity_type == "ad":
            # History: all ads in the same adset
            query = select(
                H.ad_id.label('entity_id'),
                func.sum(H.spend).label('spend'),
                func.sum(H.leads).label('leads'),
            ).where(
                and_(
                    H.config_id == config_id,
                    H.adset_id == parent_id,
                    H.ad_id != entity_id,
                    H.date >= start_date,
                )
            ).group_by(H.ad_id)

            # Today: same filter
            today_query = select(
                T.ad_id.label('entity_id'),
                func.sum(T.spend).label('spend'),
                func.sum(T.leads).label('leads'),
            ).where(
                and_(
                    T.config_id == config_id,
                    T.adset_id == parent_id,
                    T.ad_id != entity_id,
                )
            ).group_by(T.ad_id)
        else:
            return None

        # Fetch history
        result = await self.session.execute(query)
        rows = result.fetchall()

        # Fetch today
        today_result = await self.session.execute(today_query)
        today_rows = today_result.fetchall()

        if not rows and not today_rows:
            return None

        # Build history df
        dfs = []
        if rows:
            df_hist = pd.DataFrame(rows, columns=['entity_id', 'spend', 'leads'])
            df_hist['spend'] = df_hist['spend'].astype(float)
            df_hist['leads'] = df_hist['leads'].astype(int)
            dfs.append(df_hist)

        # Build today df
        if today_rows:
            df_today = pd.DataFrame(today_rows, columns=['entity_id', 'spend', 'leads'])
            df_today['spend'] = df_today['spend'].astype(float)
            df_today['leads'] = df_today['leads'].astype(int)
            dfs.append(df_today)

        # Merge: sum spend and leads by entity
        df = pd.concat(dfs, ignore_index=True)
        df = df.groupby('entity_id').agg({
            'spend': 'sum',
            'leads': 'sum',
        }).reset_index()

        df['cpl'] = df.apply(
            lambda r: r['spend'] / r['leads'] if r['leads'] > 0 else 0, axis=1
        )

        return df

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
