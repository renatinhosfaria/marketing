"""
Motor de regras para recomendações de otimização.
Gera recomendações baseadas em regras de negócio definidas.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from app.services.feature_engineering import CampaignFeatures
from app.db.models.ml_models import RecommendationType
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Recommendation:
    """Estrutura de uma recomendação."""
    recommendation_type: RecommendationType
    priority: int  # 1-10, maior = mais urgente
    title: str
    description: str
    suggested_action: dict
    confidence_score: float  # 0.0 - 1.0
    reasoning: dict
    entity_type: str
    entity_id: str
    config_id: int
    expires_in_days: int = 7


class RuleEngine:
    """
    Motor de regras para gerar recomendações.
    
    Regras implementadas:
    1. SCALE_UP: Escalar campanhas de alta performance
    2. BUDGET_DECREASE: Reduzir budget de campanhas com CPL alto
    3. PAUSE_CAMPAIGN: Pausar campanhas com baixa performance persistente
    4. CREATIVE_REFRESH: Renovar criativos quando frequência alta
    5. AUDIENCE_REVIEW: Revisar segmentação quando CPL subindo
    6. BUDGET_INCREASE: Aumentar budget moderadamente
    7. REACTIVATE: Reativar campanhas pausadas com bom histórico
    """
    
    def __init__(
        self,
        avg_cpl: float,
        avg_ctr: float,
        avg_conversion_rate: float = 0.0,
    ):
        """
        Inicializa o motor de regras com métricas de referência.
        
        Args:
            avg_cpl: CPL médio da conta (referência)
            avg_ctr: CTR médio da conta
            avg_conversion_rate: Taxa de conversão média
        """
        self.avg_cpl = avg_cpl if avg_cpl > 0 else 50.0  # Default se não houver dados
        self.avg_ctr = avg_ctr if avg_ctr > 0 else 1.0
        self.avg_conversion_rate = avg_conversion_rate if avg_conversion_rate > 0 else 5.0
        
        # Thresholds das configurações
        self.threshold_cpl_low = settings.threshold_cpl_low  # 0.7
        self.threshold_cpl_high = settings.threshold_cpl_high  # 1.3
        self.threshold_ctr_good = settings.threshold_ctr_good  # 1.2
        self.threshold_frequency_high = settings.threshold_frequency_high  # 3.0
        self.threshold_days_underperforming = settings.threshold_days_underperforming  # 7
        
        self.logger = get_logger(self.__class__.__name__)
    
    def generate_recommendations(
        self,
        features: CampaignFeatures,
        classification_tier: Optional[str] = None,
        days_in_current_tier: int = 0
    ) -> list[Recommendation]:
        """
        Gera recomendações para uma campanha baseado nas features.
        
        Args:
            features: Features da campanha
            classification_tier: Tier atual da classificação (se disponível)
            days_in_current_tier: Dias no tier atual
            
        Returns:
            Lista de recomendações ordenadas por prioridade
        """
        recommendations = []
        
        # Pular campanhas sem dados suficientes
        if features.days_active < 3:
            self.logger.debug(
                "Campanha muito nova para recomendações",
                campaign_id=features.campaign_id,
                days_active=features.days_active
            )
            return []
        
        # Regra 1: SCALE_UP - Escalar vencedores
        rec = self._check_scale_up(features, classification_tier)
        if rec:
            recommendations.append(rec)
        
        # Regra 2: BUDGET_DECREASE - CPL muito alto
        rec = self._check_budget_decrease(features)
        if rec:
            recommendations.append(rec)
        
        # Regra 3: PAUSE_CAMPAIGN - Performance ruim persistente
        rec = self._check_pause_campaign(features, classification_tier, days_in_current_tier)
        if rec:
            recommendations.append(rec)
        
        # Regra 4: CREATIVE_REFRESH - Frequência alta
        rec = self._check_creative_refresh(features)
        if rec:
            recommendations.append(rec)
        
        # Regra 5: AUDIENCE_REVIEW - CPL subindo
        rec = self._check_audience_review(features)
        if rec:
            recommendations.append(rec)
        
        # Regra 6: BUDGET_INCREASE - Performance moderada com potencial
        rec = self._check_budget_increase(features)
        if rec:
            recommendations.append(rec)
        
        # Regra 7: REACTIVATE - Campanha pausada com bom histórico
        rec = self._check_reactivate(features)
        if rec:
            recommendations.append(rec)
        
        # Ordenar por prioridade (maior primeiro)
        recommendations.sort(key=lambda r: r.priority, reverse=True)
        
        # Limitar a 3 recomendações por campanha
        return recommendations[:3]
    
    def _check_scale_up(
        self,
        features: CampaignFeatures,
        tier: Optional[str]
    ) -> Optional[Recommendation]:
        """
        Regra: Escalar campanhas de alta performance.
        
        Triggers:
        - CPL abaixo de 70% da média
        - Leads consistentes (>= 5 dias com leads em 7)
        - Tendência de CPL estável ou melhorando
        """
        if not features.is_active:
            return None
        
        cpl_ratio = features.cpl_7d / self.avg_cpl if self.avg_cpl > 0 else 1.0
        
        # Verificar se é high performer
        is_high_performer = (
            cpl_ratio <= self.threshold_cpl_low and
            features.leads_7d >= 3 and
            features.days_with_leads_7d >= 4 and
            features.cpl_trend <= 10  # CPL não subindo muito
        )
        
        if not is_high_performer:
            return None
        
        # Calcular aumento sugerido baseado na performance
        if cpl_ratio <= 0.5:
            increase_pct = 100  # CPL excelente, dobrar budget
            priority = 9
            confidence = 0.9
        elif cpl_ratio <= 0.6:
            increase_pct = 75
            priority = 8
            confidence = 0.85
        else:
            increase_pct = 50
            priority = 7
            confidence = 0.8
        
        return Recommendation(
            recommendation_type=RecommendationType.SCALE_UP,
            priority=priority,
            title=f"Escalar campanha - CPL {cpl_ratio:.0%} da média",
            description=(
                f"Esta campanha está com CPL de R$ {features.cpl_7d:.2f}, "
                f"significativamente abaixo da média de R$ {self.avg_cpl:.2f}. "
                f"Com {features.leads_7d} leads nos últimos 7 dias e performance "
                f"consistente, recomendamos aumentar o budget em {increase_pct}%."
            ),
            suggested_action={
                "field": "daily_budget",
                "change_type": "percentage",
                "change_value": increase_pct,
                "expected_impact": {
                    "additional_leads": int(features.leads_7d * (increase_pct / 100) * 0.8),
                    "estimated_cpl": features.cpl_7d * 1.1,  # Margem de segurança
                }
            },
            confidence_score=confidence,
            reasoning={
                "cpl_7d": features.cpl_7d,
                "avg_cpl": self.avg_cpl,
                "cpl_ratio": cpl_ratio,
                "leads_7d": features.leads_7d,
                "days_with_leads": features.days_with_leads_7d,
                "cpl_trend": features.cpl_trend,
                "rule": "SCALE_UP_HIGH_PERFORMER"
            },
            entity_type="campaign",
            entity_id=features.campaign_id,
            config_id=features.config_id,
            expires_in_days=7
        )
    
    def _check_budget_decrease(self, features: CampaignFeatures) -> Optional[Recommendation]:
        """
        Regra: Reduzir budget de campanhas com CPL muito alto.
        
        Triggers:
        - CPL acima de 150% da média
        - Tendência de CPL subindo
        """
        if not features.is_active or features.spend_7d < 50:
            return None
        
        cpl_ratio = features.cpl_7d / self.avg_cpl if self.avg_cpl > 0 else 1.0
        
        is_high_cpl = (
            cpl_ratio >= 1.5 and
            features.cpl_trend > 0  # CPL subindo
        )
        
        if not is_high_cpl:
            return None
        
        # Calcular redução
        if cpl_ratio >= 2.0:
            decrease_pct = 50
            priority = 8
            confidence = 0.85
        else:
            decrease_pct = 30
            priority = 6
            confidence = 0.75
        
        return Recommendation(
            recommendation_type=RecommendationType.BUDGET_DECREASE,
            priority=priority,
            title=f"Reduzir budget - CPL {cpl_ratio:.0%} da média",
            description=(
                f"O CPL desta campanha (R$ {features.cpl_7d:.2f}) está "
                f"{(cpl_ratio - 1) * 100:.0f}% acima da média e com tendência de alta "
                f"({features.cpl_trend:+.1f}%). Recomendamos reduzir o budget em "
                f"{decrease_pct}% enquanto otimiza a campanha."
            ),
            suggested_action={
                "field": "daily_budget",
                "change_type": "percentage",
                "change_value": -decrease_pct,
                "expected_impact": {
                    "spend_reduction": features.spend_7d * (decrease_pct / 100),
                }
            },
            confidence_score=confidence,
            reasoning={
                "cpl_7d": features.cpl_7d,
                "avg_cpl": self.avg_cpl,
                "cpl_ratio": cpl_ratio,
                "cpl_trend": features.cpl_trend,
                "rule": "BUDGET_DECREASE_HIGH_CPL"
            },
            entity_type="campaign",
            entity_id=features.campaign_id,
            config_id=features.config_id,
            expires_in_days=5
        )
    
    def _check_pause_campaign(
        self,
        features: CampaignFeatures,
        tier: Optional[str],
        days_in_tier: int
    ) -> Optional[Recommendation]:
        """
        Regra: Pausar campanhas com baixa performance persistente.
        
        Triggers:
        - Sem leads ou CPL muito alto por mais de 7 dias
        - Tier UNDERPERFORMER por mais de 7 dias
        - Gasto significativo sem retorno
        """
        if not features.is_active:
            return None
        
        cpl_ratio = features.cpl_7d / self.avg_cpl if self.avg_cpl > 0 else 1.0
        
        # Critérios para pausar
        no_leads = features.leads_7d == 0 and features.spend_7d > 100
        very_high_cpl = cpl_ratio >= 2.5 and features.days_active >= 7
        underperformer_long = (
            tier == "UNDERPERFORMER" and 
            days_in_tier >= self.threshold_days_underperforming
        )
        
        should_pause = no_leads or very_high_cpl or underperformer_long
        
        if not should_pause:
            return None
        
        # Determinar razão principal
        if no_leads:
            reason = f"Sem leads nos últimos 7 dias com gasto de R$ {features.spend_7d:.2f}"
            priority = 7
        elif very_high_cpl:
            reason = f"CPL de R$ {features.cpl_7d:.2f} ({cpl_ratio:.0%} da média) por 7+ dias"
            priority = 6
        else:
            reason = f"Performance baixa persistente por {days_in_tier} dias"
            priority = 5
        
        return Recommendation(
            recommendation_type=RecommendationType.PAUSE_CAMPAIGN,
            priority=priority,
            title="Pausar campanha - Performance insuficiente",
            description=(
                f"Recomendamos pausar esta campanha. {reason}. "
                f"Avalie a segmentação, criativos e landing page antes de reativar."
            ),
            suggested_action={
                "field": "status",
                "change_type": "set",
                "change_value": "PAUSED",
                "expected_impact": {
                    "spend_saved": features.spend_7d,
                }
            },
            confidence_score=0.7,
            reasoning={
                "no_leads": no_leads,
                "very_high_cpl": very_high_cpl,
                "underperformer_long": underperformer_long,
                "spend_7d": features.spend_7d,
                "leads_7d": features.leads_7d,
                "cpl_ratio": cpl_ratio,
                "rule": "PAUSE_UNDERPERFORMER"
            },
            entity_type="campaign",
            entity_id=features.campaign_id,
            config_id=features.config_id,
            expires_in_days=3
        )
    
    def _check_creative_refresh(self, features: CampaignFeatures) -> Optional[Recommendation]:
        """
        Regra: Renovar criativos quando frequência alta.
        
        Triggers:
        - Frequência > 3.0
        - CTR em queda
        - CPL subindo
        """
        if not features.is_active:
            return None
        
        high_frequency = features.frequency_7d >= self.threshold_frequency_high
        ctr_declining = features.ctr_trend < -10  # CTR caiu mais de 10%
        cpl_rising = features.cpl_trend > 15  # CPL subiu mais de 15%
        
        # Precisa de frequência alta + algum sinal de fadiga
        if not high_frequency or not (ctr_declining or cpl_rising):
            return None
        
        priority = 6 if (ctr_declining and cpl_rising) else 5
        
        return Recommendation(
            recommendation_type=RecommendationType.CREATIVE_REFRESH,
            priority=priority,
            title=f"Renovar criativos - Frequência {features.frequency_7d:.1f}",
            description=(
                f"A frequência desta campanha está em {features.frequency_7d:.1f}, "
                f"indicando que o público já viu os anúncios múltiplas vezes. "
                f"O CTR variou {features.ctr_trend:+.1f}% e o CPL {features.cpl_trend:+.1f}%. "
                f"Recomendamos criar novos criativos para combater a fadiga de anúncio."
            ),
            suggested_action={
                "action": "create_new_creatives",
                "recommendation": "Criar 2-3 novos criativos com abordagens diferentes",
                "expected_impact": {
                    "ctr_improvement": "10-20%",
                    "cpl_reduction": "10-15%",
                }
            },
            confidence_score=0.75,
            reasoning={
                "frequency_7d": features.frequency_7d,
                "ctr_trend": features.ctr_trend,
                "cpl_trend": features.cpl_trend,
                "threshold_frequency": self.threshold_frequency_high,
                "rule": "CREATIVE_REFRESH_HIGH_FREQUENCY"
            },
            entity_type="campaign",
            entity_id=features.campaign_id,
            config_id=features.config_id,
            expires_in_days=7
        )
    
    def _check_audience_review(self, features: CampaignFeatures) -> Optional[Recommendation]:
        """
        Regra: Revisar segmentação quando CPL subindo consistentemente.
        
        Triggers:
        - CPL subindo mais de 20%
        - Leads caindo
        - CTR estável ou subindo (não é problema de criativo)
        """
        if not features.is_active or features.leads_7d < 2:
            return None
        
        cpl_rising_fast = features.cpl_trend > 20
        leads_declining = features.leads_trend < -15
        ctr_not_issue = features.ctr_trend >= -5  # CTR não caiu muito
        
        if not (cpl_rising_fast and leads_declining and ctr_not_issue):
            return None
        
        return Recommendation(
            recommendation_type=RecommendationType.AUDIENCE_REVIEW,
            priority=5,
            title="Revisar segmentação - CPL em alta",
            description=(
                f"O CPL desta campanha aumentou {features.cpl_trend:.1f}% enquanto "
                f"os leads caíram {abs(features.leads_trend):.1f}%, mas o CTR permanece "
                f"estável ({features.ctr_trend:+.1f}%). Isso sugere que o problema não "
                f"está nos criativos, mas na qualidade do público. Recomendamos revisar "
                f"a segmentação e considerar exclusões ou novos públicos."
            ),
            suggested_action={
                "action": "review_targeting",
                "recommendations": [
                    "Analisar breakdown por idade/gênero/região",
                    "Verificar sobreposição de públicos",
                    "Testar exclusões de públicos que converteram",
                    "Considerar públicos lookalike atualizados"
                ]
            },
            confidence_score=0.7,
            reasoning={
                "cpl_trend": features.cpl_trend,
                "leads_trend": features.leads_trend,
                "ctr_trend": features.ctr_trend,
                "rule": "AUDIENCE_REVIEW_CPL_RISING"
            },
            entity_type="campaign",
            entity_id=features.campaign_id,
            config_id=features.config_id,
            expires_in_days=7
        )
    
    def _check_budget_increase(self, features: CampaignFeatures) -> Optional[Recommendation]:
        """
        Regra: Aumentar budget moderadamente para campanhas com boa performance.
        
        Triggers:
        - CPL entre 70-100% da média (bom, mas não excepcional)
        - Performance estável
        - Leads consistentes
        """
        if not features.is_active:
            return None
        
        cpl_ratio = features.cpl_7d / self.avg_cpl if self.avg_cpl > 0 else 1.0
        
        # Boa performance, mas não excepcional
        good_performance = (
            0.7 <= cpl_ratio <= 1.0 and
            features.leads_7d >= 2 and
            features.days_with_leads_7d >= 3 and
            abs(features.cpl_trend) <= 15  # CPL estável
        )
        
        if not good_performance:
            return None
        
        increase_pct = 25 if cpl_ratio <= 0.85 else 15
        
        return Recommendation(
            recommendation_type=RecommendationType.BUDGET_INCREASE,
            priority=4,
            title=f"Aumentar budget em {increase_pct}%",
            description=(
                f"Esta campanha está com performance sólida: CPL de R$ {features.cpl_7d:.2f} "
                f"({cpl_ratio:.0%} da média), {features.leads_7d} leads em 7 dias, "
                f"e tendência estável. Um aumento moderado de {increase_pct}% pode "
                f"gerar mais leads mantendo a eficiência."
            ),
            suggested_action={
                "field": "daily_budget",
                "change_type": "percentage",
                "change_value": increase_pct,
                "expected_impact": {
                    "additional_leads": int(features.leads_7d * (increase_pct / 100) * 0.9),
                }
            },
            confidence_score=0.7,
            reasoning={
                "cpl_7d": features.cpl_7d,
                "cpl_ratio": cpl_ratio,
                "leads_7d": features.leads_7d,
                "cpl_trend": features.cpl_trend,
                "rule": "BUDGET_INCREASE_GOOD_PERFORMER"
            },
            entity_type="campaign",
            entity_id=features.campaign_id,
            config_id=features.config_id,
            expires_in_days=7
        )
    
    def _check_reactivate(self, features: CampaignFeatures) -> Optional[Recommendation]:
        """
        Regra: Reativar campanhas pausadas com bom histórico.
        
        Triggers:
        - Campanha pausada
        - Histórico de CPL bom (30 dias)
        - Teve leads consistentes antes de pausar
        """
        if features.is_active:
            return None
        
        # Verificar histórico de 30 dias
        cpl_30d_ratio = features.cpl_30d / self.avg_cpl if self.avg_cpl > 0 else 1.0
        
        good_history = (
            cpl_30d_ratio <= 1.1 and
            features.leads_30d >= 5
        )
        
        if not good_history:
            return None
        
        return Recommendation(
            recommendation_type=RecommendationType.REACTIVATE,
            priority=3,
            title="Considerar reativar campanha",
            description=(
                f"Esta campanha está pausada mas tem histórico positivo: "
                f"CPL médio de R$ {features.cpl_30d:.2f} nos últimos 30 dias "
                f"com {features.leads_30d} leads. Considere reativar com "
                f"criativos atualizados ou pequenos ajustes de segmentação."
            ),
            suggested_action={
                "field": "status",
                "change_type": "set",
                "change_value": "ACTIVE",
                "recommendations": [
                    "Atualizar criativos antes de reativar",
                    "Começar com budget conservador",
                    "Monitorar primeiros 3 dias de perto"
                ]
            },
            confidence_score=0.6,
            reasoning={
                "cpl_30d": features.cpl_30d,
                "cpl_30d_ratio": cpl_30d_ratio,
                "leads_30d": features.leads_30d,
                "is_active": features.is_active,
                "rule": "REACTIVATE_GOOD_HISTORY"
            },
            entity_type="campaign",
            entity_id=features.campaign_id,
            config_id=features.config_id,
            expires_in_days=14
        )


def create_rule_engine(avg_metrics: dict) -> RuleEngine:
    """
    Factory para criar RuleEngine com métricas de referência.
    
    Args:
        avg_metrics: Dict com avg_cpl, avg_ctr, avg_conversion_rate
    """
    return RuleEngine(
        avg_cpl=avg_metrics.get('avg_cpl', 50.0),
        avg_ctr=avg_metrics.get('avg_ctr', 1.0),
        avg_conversion_rate=avg_metrics.get('avg_conversion_rate', 5.0),
    )
