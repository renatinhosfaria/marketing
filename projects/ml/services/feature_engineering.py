"""
Engenharia de Features para modelos de ML.
Transforma dados brutos de Facebook Ads em features utilizáveis.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from decimal import Decimal

import numpy as np
import pandas as pd

from shared.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EntityFeatures:
    """Features genericas para qualquer entidade (campaign, adset, ad)."""
    entity_type: str  # "campaign", "adset", "ad"
    entity_id: str
    config_id: int
    parent_id: Optional[str] = None  # campaign_id para adset, adset_id para ad

    # Metricas basicas (7d)
    spend_7d: float = 0.0
    impressions_7d: int = 0
    clicks_7d: int = 0
    leads_7d: int = 0

    # Metricas calculadas
    cpl_7d: float = 0.0
    ctr_7d: float = 0.0
    cpc_7d: float = 0.0
    conversion_rate_7d: float = 0.0

    # Tendencias (7d vs 7d anterior)
    cpl_trend: float = 0.0
    leads_trend: float = 0.0
    spend_trend: float = 0.0
    ctr_trend: float = 0.0

    # Metricas de periodo mais longo
    cpl_14d: float = 0.0
    leads_14d: int = 0
    cpl_30d: float = 0.0
    leads_30d: int = 0
    avg_daily_spend_30d: float = 0.0

    # Volatilidade
    cpl_std_7d: float = 0.0
    leads_std_7d: float = 0.0

    # Sazonalidade
    best_day_of_week: int = 0
    worst_day_of_week: int = 0

    # Frequencia e alcance
    frequency_7d: float = 0.0
    reach_7d: int = 0

    # Consistencia
    days_with_leads_7d: int = 0
    days_active: int = 0

    # Status
    is_active: bool = True
    has_budget: bool = True

    # Contexto hierarquico (novo para multi-level)
    share_of_parent_spend: float = 0.0  # % do gasto do parent
    share_of_parent_leads: float = 0.0  # % dos leads do parent
    performance_vs_siblings: float = 0.0  # CPL comparado com media dos irmaos

    # Timestamp
    computed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.computed_at is None:
            self.computed_at = datetime.now()


@dataclass
class CampaignFeatures:
    """Features extraídas de uma campanha."""
    campaign_id: str
    config_id: int
    
    # Métricas básicas (últimos 7 dias)
    spend_7d: float
    impressions_7d: int
    clicks_7d: int
    leads_7d: int
    
    # Métricas calculadas
    cpl_7d: float
    ctr_7d: float
    cpc_7d: float
    conversion_rate_7d: float
    
    # Tendências (7d vs 7d anterior)
    cpl_trend: float  # % mudança
    leads_trend: float
    spend_trend: float
    ctr_trend: float
    
    # Métricas de 14 dias para contexto
    cpl_14d: float
    leads_14d: int
    
    # Métricas de 30 dias
    cpl_30d: float
    leads_30d: int
    avg_daily_spend_30d: float
    
    # Volatilidade
    cpl_std_7d: float  # Desvio padrão do CPL
    leads_std_7d: float
    
    # Sazonalidade (dia da semana)
    best_day_of_week: int  # 0-6
    worst_day_of_week: int
    
    # Frequência e alcance
    frequency_7d: float
    reach_7d: int
    
    # Consistência
    days_with_leads_7d: int
    days_active: int
    
    # Status
    is_active: bool
    has_budget: bool
    
    # Timestamp
    computed_at: datetime


# Note: CampaignFeatures is kept for backward compatibility
# New code should use EntityFeatures with entity_type="campaign"


@dataclass
class MetricsSnapshot:
    """Snapshot de métricas em um ponto no tempo."""
    date: datetime
    spend: float
    impressions: int
    clicks: int
    leads: int
    cpl: float
    ctr: float
    cpc: float
    frequency: float
    reach: int


class FeatureEngineer:
    """
    Classe responsável por criar features para os modelos de ML.
    """
    
    # Constantes para cálculos
    MIN_SPEND_FOR_CPL = 1.0  # Mínimo de gasto para calcular CPL
    MIN_IMPRESSIONS_FOR_CTR = 100  # Mínimo de impressões para CTR
    MIN_CLICKS_FOR_CPC = 1  # Mínimo de cliques para CPC
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def compute_campaign_features(
        self,
        daily_data: pd.DataFrame,
        campaign_info: dict,
        reference_date: Optional[datetime] = None
    ) -> CampaignFeatures:
        """
        Calcula features de uma campanha a partir dos dados diários.
        
        Args:
            daily_data: DataFrame com colunas [date, spend, impressions, clicks, leads, ...]
            campaign_info: Dict com informações da campanha (id, config_id, status, etc)
            reference_date: Data de referência para cálculos (default: hoje)
            
        Returns:
            CampaignFeatures com todas as features calculadas
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        # Garantir que temos dados ordenados por data
        df = daily_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=False)
        
        # Filtrar por períodos
        df_7d = df[df['date'] >= reference_date - timedelta(days=7)]
        df_14d = df[df['date'] >= reference_date - timedelta(days=14)]
        df_30d = df[df['date'] >= reference_date - timedelta(days=30)]
        df_prev_7d = df[
            (df['date'] >= reference_date - timedelta(days=14)) & 
            (df['date'] < reference_date - timedelta(days=7))
        ]
        
        # Métricas básicas 7d
        spend_7d = df_7d['spend'].sum() if not df_7d.empty else 0.0
        impressions_7d = int(df_7d['impressions'].sum()) if not df_7d.empty else 0
        clicks_7d = int(df_7d['clicks'].sum()) if not df_7d.empty else 0
        leads_7d = int(df_7d['leads'].sum()) if not df_7d.empty else 0
        
        # Métricas calculadas 7d
        cpl_7d = self._safe_divide(spend_7d, leads_7d)
        ctr_7d = self._safe_divide(clicks_7d, impressions_7d) * 100
        cpc_7d = self._safe_divide(spend_7d, clicks_7d)
        conversion_rate_7d = self._safe_divide(leads_7d, clicks_7d) * 100
        
        # Métricas período anterior (para tendências)
        spend_prev = df_prev_7d['spend'].sum() if not df_prev_7d.empty else 0.0
        leads_prev = int(df_prev_7d['leads'].sum()) if not df_prev_7d.empty else 0
        cpl_prev = self._safe_divide(spend_prev, leads_prev)
        clicks_prev = int(df_prev_7d['clicks'].sum()) if not df_prev_7d.empty else 0
        impressions_prev = int(df_prev_7d['impressions'].sum()) if not df_prev_7d.empty else 0
        ctr_prev = self._safe_divide(clicks_prev, impressions_prev) * 100
        
        # Tendências (% de mudança)
        cpl_trend = self._compute_trend(cpl_7d, cpl_prev)
        leads_trend = self._compute_trend(leads_7d, leads_prev)
        spend_trend = self._compute_trend(spend_7d, spend_prev)
        ctr_trend = self._compute_trend(ctr_7d, ctr_prev)
        
        # Métricas 14d
        spend_14d = df_14d['spend'].sum() if not df_14d.empty else 0.0
        leads_14d = int(df_14d['leads'].sum()) if not df_14d.empty else 0
        cpl_14d = self._safe_divide(spend_14d, leads_14d)
        
        # Métricas 30d
        spend_30d = df_30d['spend'].sum() if not df_30d.empty else 0.0
        leads_30d = int(df_30d['leads'].sum()) if not df_30d.empty else 0
        cpl_30d = self._safe_divide(spend_30d, leads_30d)
        days_in_30d = len(df_30d) if not df_30d.empty else 1
        avg_daily_spend_30d = spend_30d / max(days_in_30d, 1)
        
        # Volatilidade (desvio padrão)
        if not df_7d.empty and len(df_7d) > 1:
            daily_cpls = df_7d.apply(
                lambda r: self._safe_divide(r['spend'], r['leads']), axis=1
            )
            cpl_std_7d = daily_cpls.std() if not daily_cpls.isna().all() else 0.0
            leads_std_7d = df_7d['leads'].std()
        else:
            cpl_std_7d = 0.0
            leads_std_7d = 0.0
        
        # Análise por dia da semana
        best_day, worst_day = self._analyze_day_of_week(df_30d)
        
        # Frequência e alcance
        frequency_7d = df_7d['frequency'].mean() if 'frequency' in df_7d.columns and not df_7d.empty else 0.0
        reach_7d = int(df_7d['reach'].sum()) if 'reach' in df_7d.columns and not df_7d.empty else 0
        
        # Consistência
        days_with_leads_7d = int((df_7d['leads'] > 0).sum()) if not df_7d.empty else 0
        days_active = len(df) if not df.empty else 0
        
        # Status
        is_active = campaign_info.get('status', '').upper() == 'ACTIVE'
        daily_budget = campaign_info.get('daily_budget', 0) or 0
        lifetime_budget = campaign_info.get('lifetime_budget', 0) or 0
        has_budget = float(daily_budget) > 0 or float(lifetime_budget) > 0
        
        return CampaignFeatures(
            campaign_id=campaign_info['campaign_id'],
            config_id=campaign_info['config_id'],
            spend_7d=float(spend_7d),
            impressions_7d=impressions_7d,
            clicks_7d=clicks_7d,
            leads_7d=leads_7d,
            cpl_7d=float(cpl_7d),
            ctr_7d=float(ctr_7d),
            cpc_7d=float(cpc_7d),
            conversion_rate_7d=float(conversion_rate_7d),
            cpl_trend=float(cpl_trend),
            leads_trend=float(leads_trend),
            spend_trend=float(spend_trend),
            ctr_trend=float(ctr_trend),
            cpl_14d=float(cpl_14d),
            leads_14d=leads_14d,
            cpl_30d=float(cpl_30d),
            leads_30d=leads_30d,
            avg_daily_spend_30d=float(avg_daily_spend_30d),
            cpl_std_7d=float(cpl_std_7d) if not np.isnan(cpl_std_7d) else 0.0,
            leads_std_7d=float(leads_std_7d) if not np.isnan(leads_std_7d) else 0.0,
            best_day_of_week=best_day,
            worst_day_of_week=worst_day,
            frequency_7d=float(frequency_7d) if not np.isnan(frequency_7d) else 0.0,
            reach_7d=reach_7d,
            days_with_leads_7d=days_with_leads_7d,
            days_active=days_active,
            is_active=is_active,
            has_budget=has_budget,
            computed_at=datetime.now()
        )

    def compute_entity_features(
        self,
        daily_data: pd.DataFrame,
        entity_info: dict,
        entity_type: str = "campaign",
        parent_metrics: Optional[dict] = None,
        sibling_metrics: Optional[pd.DataFrame] = None,
        reference_date: Optional[datetime] = None
    ) -> EntityFeatures:
        """
        Calcula features de uma entidade a partir dos dados diarios.

        Args:
            daily_data: DataFrame com dados diarios da entidade
            entity_info: Dict com informacoes da entidade (id, config_id, status, etc)
            entity_type: 'campaign', 'adset', ou 'ad'
            parent_metrics: Dict com metricas do parent (para calcular share)
            sibling_metrics: DataFrame com metricas dos irmaos (para comparacao)
            reference_date: Data de referencia para calculos
        """
        if reference_date is None:
            reference_date = datetime.now()

        # Determine entity_id field based on type
        entity_id_field = f"{entity_type}_id"
        entity_id = entity_info.get(entity_id_field) or entity_info.get('entity_id')

        # Determine parent_id based on type
        parent_id = None
        if entity_type == "adset":
            parent_id = entity_info.get("campaign_id")
        elif entity_type == "ad":
            parent_id = entity_info.get("adset_id")

        # Use existing computation logic
        df = daily_data.copy()
        if df.empty:
            return EntityFeatures(
                entity_type=entity_type,
                entity_id=entity_id,
                config_id=entity_info['config_id'],
                parent_id=parent_id,
                is_active=entity_info.get('status', '').upper() == 'ACTIVE',
            )

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=False)

        # Filter by periods
        df_7d = df[df['date'] >= reference_date - timedelta(days=7)]
        df_14d = df[df['date'] >= reference_date - timedelta(days=14)]
        df_30d = df[df['date'] >= reference_date - timedelta(days=30)]
        df_prev_7d = df[
            (df['date'] >= reference_date - timedelta(days=14)) &
            (df['date'] < reference_date - timedelta(days=7))
        ]

        # Basic metrics 7d
        spend_7d = df_7d['spend'].sum() if not df_7d.empty else 0.0
        impressions_7d = int(df_7d['impressions'].sum()) if not df_7d.empty else 0
        clicks_7d = int(df_7d['clicks'].sum()) if not df_7d.empty else 0
        leads_7d = int(df_7d['leads'].sum()) if not df_7d.empty else 0

        # Calculated metrics
        cpl_7d = self._safe_divide(spend_7d, leads_7d)
        ctr_7d = self._safe_divide(clicks_7d, impressions_7d) * 100
        cpc_7d = self._safe_divide(spend_7d, clicks_7d)
        conversion_rate_7d = self._safe_divide(leads_7d, clicks_7d) * 100

        # Previous period metrics
        spend_prev = df_prev_7d['spend'].sum() if not df_prev_7d.empty else 0.0
        leads_prev = int(df_prev_7d['leads'].sum()) if not df_prev_7d.empty else 0
        cpl_prev = self._safe_divide(spend_prev, leads_prev)
        clicks_prev = int(df_prev_7d['clicks'].sum()) if not df_prev_7d.empty else 0
        impressions_prev = int(df_prev_7d['impressions'].sum()) if not df_prev_7d.empty else 0
        ctr_prev = self._safe_divide(clicks_prev, impressions_prev) * 100

        # Trends
        cpl_trend = self._compute_trend(cpl_7d, cpl_prev)
        leads_trend = self._compute_trend(leads_7d, leads_prev)
        spend_trend = self._compute_trend(spend_7d, spend_prev)
        ctr_trend = self._compute_trend(ctr_7d, ctr_prev)

        # 14d metrics
        spend_14d = df_14d['spend'].sum() if not df_14d.empty else 0.0
        leads_14d = int(df_14d['leads'].sum()) if not df_14d.empty else 0
        cpl_14d = self._safe_divide(spend_14d, leads_14d)

        # 30d metrics
        spend_30d = df_30d['spend'].sum() if not df_30d.empty else 0.0
        leads_30d = int(df_30d['leads'].sum()) if not df_30d.empty else 0
        cpl_30d = self._safe_divide(spend_30d, leads_30d)
        days_in_30d = len(df_30d) if not df_30d.empty else 1
        avg_daily_spend_30d = spend_30d / max(days_in_30d, 1)

        # Volatility
        if not df_7d.empty and len(df_7d) > 1:
            daily_cpls = df_7d.apply(
                lambda r: self._safe_divide(r['spend'], r['leads']), axis=1
            )
            cpl_std_7d = daily_cpls.std() if not daily_cpls.isna().all() else 0.0
            leads_std_7d = df_7d['leads'].std()
        else:
            cpl_std_7d = 0.0
            leads_std_7d = 0.0

        # Day of week analysis
        best_day, worst_day = self._analyze_day_of_week(df_30d)

        # Frequency and reach
        frequency_7d = df_7d['frequency'].mean() if 'frequency' in df_7d.columns and not df_7d.empty else 0.0
        reach_7d = int(df_7d['reach'].sum()) if 'reach' in df_7d.columns and not df_7d.empty else 0

        # Consistency
        days_with_leads_7d = int((df_7d['leads'] > 0).sum()) if not df_7d.empty else 0
        days_active = len(df) if not df.empty else 0

        # Status
        is_active = entity_info.get('status', '').upper() == 'ACTIVE'
        daily_budget = entity_info.get('daily_budget', 0) or 0
        lifetime_budget = entity_info.get('lifetime_budget', 0) or 0
        has_budget = float(daily_budget) > 0 or float(lifetime_budget) > 0

        # Hierarchical context
        share_of_parent_spend = 0.0
        share_of_parent_leads = 0.0
        if parent_metrics and parent_metrics.get('spend', 0) > 0:
            share_of_parent_spend = (spend_7d / parent_metrics['spend']) * 100
        if parent_metrics and parent_metrics.get('leads', 0) > 0:
            share_of_parent_leads = (leads_7d / parent_metrics['leads']) * 100

        performance_vs_siblings = 0.0
        if sibling_metrics is not None and not sibling_metrics.empty and 'cpl' in sibling_metrics.columns:
            avg_sibling_cpl = sibling_metrics['cpl'].mean()
            if avg_sibling_cpl > 0:
                performance_vs_siblings = ((avg_sibling_cpl - cpl_7d) / avg_sibling_cpl) * 100

        return EntityFeatures(
            entity_type=entity_type,
            entity_id=entity_id,
            config_id=entity_info['config_id'],
            parent_id=parent_id,
            spend_7d=float(spend_7d),
            impressions_7d=impressions_7d,
            clicks_7d=clicks_7d,
            leads_7d=leads_7d,
            cpl_7d=float(cpl_7d),
            ctr_7d=float(ctr_7d),
            cpc_7d=float(cpc_7d),
            conversion_rate_7d=float(conversion_rate_7d),
            cpl_trend=float(cpl_trend),
            leads_trend=float(leads_trend),
            spend_trend=float(spend_trend),
            ctr_trend=float(ctr_trend),
            cpl_14d=float(cpl_14d),
            leads_14d=leads_14d,
            cpl_30d=float(cpl_30d),
            leads_30d=leads_30d,
            avg_daily_spend_30d=float(avg_daily_spend_30d),
            cpl_std_7d=float(cpl_std_7d) if not np.isnan(cpl_std_7d) else 0.0,
            leads_std_7d=float(leads_std_7d) if not np.isnan(leads_std_7d) else 0.0,
            best_day_of_week=best_day,
            worst_day_of_week=worst_day,
            frequency_7d=float(frequency_7d) if not np.isnan(frequency_7d) else 0.0,
            reach_7d=reach_7d,
            days_with_leads_7d=days_with_leads_7d,
            days_active=days_active,
            is_active=is_active,
            has_budget=has_budget,
            share_of_parent_spend=float(share_of_parent_spend),
            share_of_parent_leads=float(share_of_parent_leads),
            performance_vs_siblings=float(performance_vs_siblings),
        )

    def compute_aggregated_metrics(
        self,
        daily_data: pd.DataFrame,
        days: int = 7
    ) -> dict:
        """
        Calcula métricas agregadas de um conjunto de campanhas.
        Usado para calcular médias de referência.
        
        Args:
            daily_data: DataFrame com dados de todas as campanhas
            days: Número de dias para agregar
            
        Returns:
            Dict com métricas agregadas
        """
        df = daily_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df['date'] >= cutoff]
        
        if df.empty:
            return {
                'avg_cpl': 0.0,
                'avg_ctr': 0.0,
                'avg_cpc': 0.0,
                'avg_conversion_rate': 0.0,
                'total_spend': 0.0,
                'total_leads': 0,
                'campaigns_count': 0
            }
        
        total_spend = df['spend'].sum()
        total_leads = df['leads'].sum()
        total_clicks = df['clicks'].sum()
        total_impressions = df['impressions'].sum()
        
        return {
            'avg_cpl': self._safe_divide(total_spend, total_leads),
            'avg_ctr': self._safe_divide(total_clicks, total_impressions) * 100,
            'avg_cpc': self._safe_divide(total_spend, total_clicks),
            'avg_conversion_rate': self._safe_divide(total_leads, total_clicks) * 100,
            'total_spend': float(total_spend),
            'total_leads': int(total_leads),
            'campaigns_count': df['campaign_id'].nunique() if 'campaign_id' in df.columns else 0
        }
    
    def features_to_dataframe(self, features: CampaignFeatures) -> pd.DataFrame:
        """Converte CampaignFeatures para DataFrame de uma linha."""
        return pd.DataFrame([{
            'campaign_id': features.campaign_id,
            'config_id': features.config_id,
            'spend_7d': features.spend_7d,
            'impressions_7d': features.impressions_7d,
            'clicks_7d': features.clicks_7d,
            'leads_7d': features.leads_7d,
            'cpl_7d': features.cpl_7d,
            'ctr_7d': features.ctr_7d,
            'cpc_7d': features.cpc_7d,
            'conversion_rate_7d': features.conversion_rate_7d,
            'cpl_trend': features.cpl_trend,
            'leads_trend': features.leads_trend,
            'spend_trend': features.spend_trend,
            'ctr_trend': features.ctr_trend,
            'cpl_14d': features.cpl_14d,
            'leads_14d': features.leads_14d,
            'cpl_30d': features.cpl_30d,
            'leads_30d': features.leads_30d,
            'avg_daily_spend_30d': features.avg_daily_spend_30d,
            'cpl_std_7d': features.cpl_std_7d,
            'leads_std_7d': features.leads_std_7d,
            'frequency_7d': features.frequency_7d,
            'reach_7d': features.reach_7d,
            'days_with_leads_7d': features.days_with_leads_7d,
            'days_active': features.days_active,
            'is_active': int(features.is_active),
            'has_budget': int(features.has_budget),
        }])
    
    def get_classification_features(self) -> list[str]:
        """Retorna lista de features usadas para classificação."""
        return [
            'cpl_7d',
            'ctr_7d',
            'conversion_rate_7d',
            'cpl_trend',
            'leads_trend',
            'cpl_std_7d',
            'leads_7d',
            'days_with_leads_7d',
            'frequency_7d',
            'spend_7d',
        ]
    
    def get_recommendation_features(self) -> list[str]:
        """Retorna lista de features usadas para recomendações."""
        return [
            'cpl_7d',
            'cpl_14d',
            'cpl_30d',
            'cpl_trend',
            'ctr_7d',
            'ctr_trend',
            'leads_7d',
            'leads_trend',
            'frequency_7d',
            'spend_7d',
            'spend_trend',
            'days_with_leads_7d',
            'conversion_rate_7d',
            'is_active',
            'has_budget',
        ]
    
    @staticmethod
    def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Divisão segura que retorna default se denominador for 0."""
        if denominator == 0 or pd.isna(denominator):
            return default
        result = numerator / denominator
        return result if not np.isnan(result) and not np.isinf(result) else default
    
    @staticmethod
    def _compute_trend(current: float, previous: float) -> float:
        """
        Calcula tendência como % de mudança.
        Positivo = melhoria (para métricas onde menor é melhor, inverta o sinal)
        """
        if previous == 0:
            return 0.0 if current == 0 else 100.0
        change = ((current - previous) / previous) * 100
        return min(max(change, -100.0), 100.0)  # Limitar entre -100% e 100%
    
    def _analyze_day_of_week(self, df: pd.DataFrame) -> tuple[int, int]:
        """
        Analisa performance por dia da semana.
        Retorna (melhor_dia, pior_dia) baseado em CPL.
        """
        if df.empty or 'date' not in df.columns:
            return (0, 0)
        
        df = df.copy()
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        
        # Agregar por dia da semana
        by_day = df.groupby('day_of_week').agg({
            'spend': 'sum',
            'leads': 'sum'
        }).reset_index()
        
        # Calcular CPL por dia
        by_day['cpl'] = by_day.apply(
            lambda r: self._safe_divide(r['spend'], r['leads'], default=float('inf')),
            axis=1
        )
        
        # Filtrar dias com dados válidos
        valid_days = by_day[by_day['leads'] > 0]
        
        if valid_days.empty:
            return (0, 0)
        
        best_day = int(valid_days.loc[valid_days['cpl'].idxmin(), 'day_of_week'])
        worst_day = int(valid_days.loc[valid_days['cpl'].idxmax(), 'day_of_week'])
        
        return (best_day, worst_day)


# Instância global
feature_engineer = FeatureEngineer()
