"""
Motor de regras para recomendações de otimização.
Gera recomendações baseadas em regras de negócio definidas.
Suporta múltiplos níveis: campaign, adset, ad.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Union

from projects.ml.services.feature_engineering import CampaignFeatures, EntityFeatures
from projects.ml.db.models import RecommendationType
from shared.config import settings
from shared.core.logging import get_logger

logger = get_logger(__name__)

# Valid entity types
VALID_ENTITY_TYPES = {"campaign", "adset", "ad"}


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
    Suporta múltiplos níveis: campaign, adset, ad.

    Regras genéricas (todos os níveis):
    1. SCALE_UP: Escalar entidades de alta performance
    2. BUDGET_DECREASE: Reduzir budget de entidades com CPL alto
    3. PAUSE: Pausar entidades com baixa performance persistente
    4. BUDGET_INCREASE: Aumentar budget moderadamente
    5. REACTIVATE: Reativar entidades pausadas com bom histórico

    Regras específicas para Adset:
    6. AUDIENCE_REVIEW: Revisar segmentação quando CPL subindo
    7. AUDIENCE_EXPANSION: Expandir audiência quando performance boa mas volume baixo
    8. AUDIENCE_NARROWING: Reduzir audiência quando CPL alto mas CTR bom

    Regras específicas para Ad:
    9. CREATIVE_REFRESH: Renovar criativos quando frequência alta
    10. CREATIVE_TEST: Sugerir teste A/B de criativos
    11. CREATIVE_WINNER: Identificar criativo vencedor para escalar
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
        features: Union[CampaignFeatures, EntityFeatures],
        classification_tier: Optional[str] = None,
        days_in_current_tier: int = 0,
        entity_type: str = "campaign",
        sibling_features: Optional[list] = None
    ) -> list[Recommendation]:
        """
        Gera recomendações para uma entidade baseado nas features.

        Args:
            features: Features da entidade (CampaignFeatures ou EntityFeatures)
            classification_tier: Tier atual da classificação (se disponível)
            days_in_current_tier: Dias no tier atual
            entity_type: Tipo de entidade ("campaign", "adset", "ad")
            sibling_features: Features de entidades irmãs (para comparação)

        Returns:
            Lista de recomendações ordenadas por prioridade
        """
        if entity_type not in VALID_ENTITY_TYPES:
            raise ValueError(f"entity_type deve ser um de: {VALID_ENTITY_TYPES}")

        recommendations = []

        # Extrair info da entidade
        entity_id = self._get_entity_id(features, entity_type)
        config_id = features.config_id
        days_active = getattr(features, 'days_active', 7)

        # Pular entidades sem dados suficientes
        if days_active < 3:
            self.logger.debug(
                f"{entity_type.capitalize()} muito nova para recomendações",
                entity_id=entity_id,
                days_active=days_active
            )
            return []

        # Regras genéricas (todos os níveis)
        rec = self._check_scale_up(features, classification_tier, entity_type)
        if rec:
            recommendations.append(rec)

        rec = self._check_budget_decrease(features, entity_type)
        if rec:
            recommendations.append(rec)

        rec = self._check_pause(features, classification_tier, days_in_current_tier, entity_type)
        if rec:
            recommendations.append(rec)

        rec = self._check_budget_increase(features, entity_type)
        if rec:
            recommendations.append(rec)

        rec = self._check_reactivate(features, entity_type)
        if rec:
            recommendations.append(rec)

        # Regras específicas por nível
        if entity_type == "adset":
            rec = self._check_audience_review(features, entity_type)
            if rec:
                recommendations.append(rec)

            rec = self._check_audience_expansion(features)
            if rec:
                recommendations.append(rec)

            rec = self._check_audience_narrowing(features)
            if rec:
                recommendations.append(rec)

        elif entity_type == "ad":
            rec = self._check_creative_refresh(features, entity_type)
            if rec:
                recommendations.append(rec)

            rec = self._check_creative_test(features)
            if rec:
                recommendations.append(rec)

            rec = self._check_creative_winner(features, sibling_features)
            if rec:
                recommendations.append(rec)

        elif entity_type == "campaign":
            # Campanhas podem ter ambas as regras de criativo e audiência
            rec = self._check_creative_refresh(features, entity_type)
            if rec:
                recommendations.append(rec)

            rec = self._check_audience_review(features, entity_type)
            if rec:
                recommendations.append(rec)

        # Ordenar por prioridade (maior primeiro)
        recommendations.sort(key=lambda r: r.priority, reverse=True)

        # Limitar a 3 recomendações por entidade
        return recommendations[:3]

    def _get_entity_id(self, features, entity_type: str) -> str:
        """Extrai o ID da entidade a partir das features."""
        if isinstance(features, EntityFeatures):
            return features.entity_id
        elif isinstance(features, CampaignFeatures):
            return features.campaign_id
        # Fallback
        return getattr(features, 'entity_id', getattr(features, 'campaign_id', ''))
    
    def _check_scale_up(
        self,
        features: Union[CampaignFeatures, EntityFeatures],
        tier: Optional[str],
        entity_type: str = "campaign"
    ) -> Optional[Recommendation]:
        """
        Regra: Escalar entidades de alta performance.

        Triggers:
        - CPL abaixo de 70% da média
        - Leads consistentes (>= 4 dias com leads em 7)
        - Tendência de CPL estável ou melhorando
        """
        is_active = getattr(features, 'is_active', True)
        if not is_active:
            return None

        cpl_7d = getattr(features, 'cpl_7d', 0)
        leads_7d = getattr(features, 'leads_7d', 0)
        days_with_leads_7d = getattr(features, 'days_with_leads_7d', 0)
        cpl_trend = getattr(features, 'cpl_trend', 0)

        cpl_ratio = cpl_7d / self.avg_cpl if self.avg_cpl > 0 else 1.0

        # Verificar se é high performer
        is_high_performer = (
            cpl_ratio <= self.threshold_cpl_low and
            leads_7d >= 3 and
            days_with_leads_7d >= 4 and
            cpl_trend <= 10  # CPL não subindo muito
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

        entity_id = self._get_entity_id(features, entity_type)
        entity_label = {"campaign": "campanha", "adset": "conjunto de anúncios", "ad": "anúncio"}

        return Recommendation(
            recommendation_type=RecommendationType.SCALE_UP,
            priority=priority,
            title=f"Escalar {entity_label.get(entity_type, entity_type)} - CPL {cpl_ratio:.0%} da média",
            description=(
                f"Este(a) {entity_label.get(entity_type, entity_type)} está com CPL de R$ {cpl_7d:.2f}, "
                f"significativamente abaixo da média de R$ {self.avg_cpl:.2f}. "
                f"Com {leads_7d} leads nos últimos 7 dias e performance "
                f"consistente, recomendamos aumentar o budget em {increase_pct}%."
            ),
            suggested_action={
                "field": "daily_budget",
                "change_type": "percentage",
                "change_value": increase_pct,
                "expected_impact": {
                    "additional_leads": int(leads_7d * (increase_pct / 100) * 0.8),
                    "estimated_cpl": cpl_7d * 1.1,  # Margem de segurança
                }
            },
            confidence_score=confidence,
            reasoning={
                "cpl_7d": cpl_7d,
                "avg_cpl": self.avg_cpl,
                "cpl_ratio": cpl_ratio,
                "leads_7d": leads_7d,
                "days_with_leads": days_with_leads_7d,
                "cpl_trend": cpl_trend,
                "rule": "SCALE_UP_HIGH_PERFORMER"
            },
            entity_type=entity_type,
            entity_id=entity_id,
            config_id=features.config_id,
            expires_in_days=7
        )
    
    def _check_budget_decrease(
        self,
        features: Union[CampaignFeatures, EntityFeatures],
        entity_type: str = "campaign"
    ) -> Optional[Recommendation]:
        """
        Regra: Reduzir budget de entidades com CPL muito alto.

        Triggers:
        - CPL acima de 150% da média
        - Tendência de CPL subindo
        """
        is_active = getattr(features, 'is_active', True)
        spend_7d = getattr(features, 'spend_7d', 0)
        cpl_7d = getattr(features, 'cpl_7d', 0)
        cpl_trend = getattr(features, 'cpl_trend', 0)

        if not is_active or spend_7d < 50:
            return None

        cpl_ratio = cpl_7d / self.avg_cpl if self.avg_cpl > 0 else 1.0

        is_high_cpl = (
            cpl_ratio >= 1.5 and
            cpl_trend > 0  # CPL subindo
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

        entity_id = self._get_entity_id(features, entity_type)
        entity_label = {"campaign": "campanha", "adset": "conjunto de anúncios", "ad": "anúncio"}

        return Recommendation(
            recommendation_type=RecommendationType.BUDGET_DECREASE,
            priority=priority,
            title=f"Reduzir budget - CPL {cpl_ratio:.0%} da média",
            description=(
                f"O CPL deste(a) {entity_label.get(entity_type, entity_type)} (R$ {cpl_7d:.2f}) está "
                f"{(cpl_ratio - 1) * 100:.0f}% acima da média e com tendência de alta "
                f"({cpl_trend:+.1f}%). Recomendamos reduzir o budget em "
                f"{decrease_pct}% enquanto otimiza."
            ),
            suggested_action={
                "field": "daily_budget",
                "change_type": "percentage",
                "change_value": -decrease_pct,
                "expected_impact": {
                    "spend_reduction": spend_7d * (decrease_pct / 100),
                }
            },
            confidence_score=confidence,
            reasoning={
                "cpl_7d": cpl_7d,
                "avg_cpl": self.avg_cpl,
                "cpl_ratio": cpl_ratio,
                "cpl_trend": cpl_trend,
                "rule": "BUDGET_DECREASE_HIGH_CPL"
            },
            entity_type=entity_type,
            entity_id=entity_id,
            config_id=features.config_id,
            expires_in_days=5
        )
    
    def _check_pause(
        self,
        features: Union[CampaignFeatures, EntityFeatures],
        tier: Optional[str],
        days_in_tier: int,
        entity_type: str = "campaign"
    ) -> Optional[Recommendation]:
        """
        Regra: Pausar entidades com baixa performance persistente.

        Triggers:
        - Sem leads ou CPL muito alto por mais de 7 dias
        - Tier UNDERPERFORMER por mais de 7 dias
        - Gasto significativo sem retorno
        """
        is_active = getattr(features, 'is_active', True)
        if not is_active:
            return None

        cpl_7d = getattr(features, 'cpl_7d', 0)
        spend_7d = getattr(features, 'spend_7d', 0)
        leads_7d = getattr(features, 'leads_7d', 0)
        days_active = getattr(features, 'days_active', 7)

        cpl_ratio = cpl_7d / self.avg_cpl if self.avg_cpl > 0 else 1.0

        # Critérios para pausar
        no_leads = leads_7d == 0 and spend_7d > 100
        very_high_cpl = cpl_ratio >= 2.5 and days_active >= 7
        underperformer_long = (
            tier == "UNDERPERFORMER" and
            days_in_tier >= self.threshold_days_underperforming
        )

        should_pause = no_leads or very_high_cpl or underperformer_long

        if not should_pause:
            return None

        entity_id = self._get_entity_id(features, entity_type)
        entity_label = {"campaign": "campanha", "adset": "conjunto de anúncios", "ad": "anúncio"}

        # Determinar razão principal
        if no_leads:
            reason = f"Sem leads nos últimos 7 dias com gasto de R$ {spend_7d:.2f}"
            priority = 7
        elif very_high_cpl:
            reason = f"CPL de R$ {cpl_7d:.2f} ({cpl_ratio:.0%} da média) por 7+ dias"
            priority = 6
        else:
            reason = f"Performance baixa persistente por {days_in_tier} dias"
            priority = 5

        return Recommendation(
            recommendation_type=RecommendationType.PAUSE,
            priority=priority,
            title=f"Pausar {entity_label.get(entity_type, entity_type)} - Performance insuficiente",
            description=(
                f"Recomendamos pausar este(a) {entity_label.get(entity_type, entity_type)}. {reason}. "
                f"Avalie a segmentação, criativos e landing page antes de reativar."
            ),
            suggested_action={
                "field": "status",
                "change_type": "set",
                "change_value": "PAUSED",
                "expected_impact": {
                    "spend_saved": spend_7d,
                }
            },
            confidence_score=0.7,
            reasoning={
                "no_leads": no_leads,
                "very_high_cpl": very_high_cpl,
                "underperformer_long": underperformer_long,
                "spend_7d": spend_7d,
                "leads_7d": leads_7d,
                "cpl_ratio": cpl_ratio,
                "rule": "PAUSE_UNDERPERFORMER"
            },
            entity_type=entity_type,
            entity_id=entity_id,
            config_id=features.config_id,
            expires_in_days=3
        )
    
    def _check_creative_refresh(
        self,
        features: Union[CampaignFeatures, EntityFeatures],
        entity_type: str = "campaign"
    ) -> Optional[Recommendation]:
        """
        Regra: Renovar criativos quando frequência alta.
        Aplicável para campaigns e ads.

        Triggers:
        - Frequência > 3.0
        - CTR em queda
        - CPL subindo
        """
        is_active = getattr(features, 'is_active', True)
        if not is_active:
            return None

        frequency_7d = getattr(features, 'frequency_7d', 0)
        ctr_trend = getattr(features, 'ctr_trend', 0)
        cpl_trend = getattr(features, 'cpl_trend', 0)

        high_frequency = frequency_7d >= self.threshold_frequency_high
        ctr_declining = ctr_trend < -10  # CTR caiu mais de 10%
        cpl_rising = cpl_trend > 15  # CPL subiu mais de 15%

        # Precisa de frequência alta + algum sinal de fadiga
        if not high_frequency or not (ctr_declining or cpl_rising):
            return None

        entity_id = self._get_entity_id(features, entity_type)
        entity_label = {"campaign": "campanha", "adset": "conjunto de anúncios", "ad": "anúncio"}

        priority = 6 if (ctr_declining and cpl_rising) else 5

        return Recommendation(
            recommendation_type=RecommendationType.CREATIVE_REFRESH,
            priority=priority,
            title=f"Renovar criativos - Frequência {frequency_7d:.1f}",
            description=(
                f"A frequência deste(a) {entity_label.get(entity_type, entity_type)} está em {frequency_7d:.1f}, "
                f"indicando que o público já viu os anúncios múltiplas vezes. "
                f"O CTR variou {ctr_trend:+.1f}% e o CPL {cpl_trend:+.1f}%. "
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
                "frequency_7d": frequency_7d,
                "ctr_trend": ctr_trend,
                "cpl_trend": cpl_trend,
                "threshold_frequency": self.threshold_frequency_high,
                "rule": "CREATIVE_REFRESH_HIGH_FREQUENCY"
            },
            entity_type=entity_type,
            entity_id=entity_id,
            config_id=features.config_id,
            expires_in_days=7
        )
    
    def _check_audience_review(
        self,
        features: Union[CampaignFeatures, EntityFeatures],
        entity_type: str = "campaign"
    ) -> Optional[Recommendation]:
        """
        Regra: Revisar segmentação quando CPL subindo consistentemente.
        Aplicável para campaigns e adsets.

        Triggers:
        - CPL subindo mais de 20%
        - Leads caindo
        - CTR estável ou subindo (não é problema de criativo)
        """
        is_active = getattr(features, 'is_active', True)
        leads_7d = getattr(features, 'leads_7d', 0)
        cpl_trend = getattr(features, 'cpl_trend', 0)
        leads_trend = getattr(features, 'leads_trend', 0)
        ctr_trend = getattr(features, 'ctr_trend', 0)

        if not is_active or leads_7d < 2:
            return None

        cpl_rising_fast = cpl_trend > 20
        leads_declining = leads_trend < -15
        ctr_not_issue = ctr_trend >= -5  # CTR não caiu muito

        if not (cpl_rising_fast and leads_declining and ctr_not_issue):
            return None

        entity_id = self._get_entity_id(features, entity_type)
        entity_label = {"campaign": "campanha", "adset": "conjunto de anúncios", "ad": "anúncio"}

        return Recommendation(
            recommendation_type=RecommendationType.AUDIENCE_REVIEW,
            priority=5,
            title="Revisar segmentação - CPL em alta",
            description=(
                f"O CPL deste(a) {entity_label.get(entity_type, entity_type)} aumentou {cpl_trend:.1f}% enquanto "
                f"os leads caíram {abs(leads_trend):.1f}%, mas o CTR permanece "
                f"estável ({ctr_trend:+.1f}%). Isso sugere que o problema não "
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
                "cpl_trend": cpl_trend,
                "leads_trend": leads_trend,
                "ctr_trend": ctr_trend,
                "rule": "AUDIENCE_REVIEW_CPL_RISING"
            },
            entity_type=entity_type,
            entity_id=entity_id,
            config_id=features.config_id,
            expires_in_days=7
        )
    
    def _check_budget_increase(
        self,
        features: Union[CampaignFeatures, EntityFeatures],
        entity_type: str = "campaign"
    ) -> Optional[Recommendation]:
        """
        Regra: Aumentar budget moderadamente para entidades com boa performance.

        Triggers:
        - CPL entre 70-100% da média (bom, mas não excepcional)
        - Performance estável
        - Leads consistentes
        """
        is_active = getattr(features, 'is_active', True)
        if not is_active:
            return None

        cpl_7d = getattr(features, 'cpl_7d', 0)
        leads_7d = getattr(features, 'leads_7d', 0)
        days_with_leads_7d = getattr(features, 'days_with_leads_7d', 0)
        cpl_trend = getattr(features, 'cpl_trend', 0)

        cpl_ratio = cpl_7d / self.avg_cpl if self.avg_cpl > 0 else 1.0

        # Boa performance, mas não excepcional
        good_performance = (
            0.7 <= cpl_ratio <= 1.0 and
            leads_7d >= 2 and
            days_with_leads_7d >= 3 and
            abs(cpl_trend) <= 15  # CPL estável
        )

        if not good_performance:
            return None

        entity_id = self._get_entity_id(features, entity_type)
        entity_label = {"campaign": "campanha", "adset": "conjunto de anúncios", "ad": "anúncio"}

        increase_pct = 25 if cpl_ratio <= 0.85 else 15

        return Recommendation(
            recommendation_type=RecommendationType.BUDGET_INCREASE,
            priority=4,
            title=f"Aumentar budget em {increase_pct}%",
            description=(
                f"Este(a) {entity_label.get(entity_type, entity_type)} está com performance sólida: CPL de R$ {cpl_7d:.2f} "
                f"({cpl_ratio:.0%} da média), {leads_7d} leads em 7 dias, "
                f"e tendência estável. Um aumento moderado de {increase_pct}% pode "
                f"gerar mais leads mantendo a eficiência."
            ),
            suggested_action={
                "field": "daily_budget",
                "change_type": "percentage",
                "change_value": increase_pct,
                "expected_impact": {
                    "additional_leads": int(leads_7d * (increase_pct / 100) * 0.9),
                }
            },
            confidence_score=0.7,
            reasoning={
                "cpl_7d": cpl_7d,
                "cpl_ratio": cpl_ratio,
                "leads_7d": leads_7d,
                "cpl_trend": cpl_trend,
                "rule": "BUDGET_INCREASE_GOOD_PERFORMER"
            },
            entity_type=entity_type,
            entity_id=entity_id,
            config_id=features.config_id,
            expires_in_days=7
        )
    
    def _check_reactivate(
        self,
        features: Union[CampaignFeatures, EntityFeatures],
        entity_type: str = "campaign"
    ) -> Optional[Recommendation]:
        """
        Regra: Reativar entidades pausadas com bom histórico.

        Triggers:
        - Entidade pausada
        - Histórico de CPL bom (30 dias)
        - Teve leads consistentes antes de pausar
        """
        is_active = getattr(features, 'is_active', True)
        if is_active:
            return None

        cpl_30d = getattr(features, 'cpl_30d', 0)
        leads_30d = getattr(features, 'leads_30d', 0)

        # Verificar histórico de 30 dias
        cpl_30d_ratio = cpl_30d / self.avg_cpl if self.avg_cpl > 0 else 1.0

        good_history = (
            cpl_30d_ratio <= 1.1 and
            leads_30d >= 5
        )

        if not good_history:
            return None

        entity_id = self._get_entity_id(features, entity_type)
        entity_label = {"campaign": "campanha", "adset": "conjunto de anúncios", "ad": "anúncio"}

        return Recommendation(
            recommendation_type=RecommendationType.REACTIVATE,
            priority=3,
            title=f"Considerar reativar {entity_label.get(entity_type, entity_type)}",
            description=(
                f"Este(a) {entity_label.get(entity_type, entity_type)} está pausado(a) mas tem histórico positivo: "
                f"CPL médio de R$ {cpl_30d:.2f} nos últimos 30 dias "
                f"com {leads_30d} leads. Considere reativar com "
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
                "cpl_30d": cpl_30d,
                "cpl_30d_ratio": cpl_30d_ratio,
                "leads_30d": leads_30d,
                "is_active": is_active,
                "rule": "REACTIVATE_GOOD_HISTORY"
            },
            entity_type=entity_type,
            entity_id=entity_id,
            config_id=features.config_id,
            expires_in_days=14
        )

    # ==================== ADSET-SPECIFIC RULES ====================

    def _check_audience_expansion(
        self,
        features: Union[CampaignFeatures, EntityFeatures]
    ) -> Optional[Recommendation]:
        """
        Regra: Expandir audiência quando performance boa mas volume baixo.
        Específico para adsets.

        Triggers:
        - CPL bom (abaixo da média)
        - Volume de leads baixo
        - Share do parent spend baixo (há espaço para crescer)
        """
        is_active = getattr(features, 'is_active', True)
        if not is_active:
            return None

        cpl_7d = getattr(features, 'cpl_7d', 0)
        leads_7d = getattr(features, 'leads_7d', 0)
        share_of_parent_spend = getattr(features, 'share_of_parent_spend', 0.5)

        cpl_ratio = cpl_7d / self.avg_cpl if self.avg_cpl > 0 else 1.0

        # CPL bom, mas volume baixo e espaço para crescer
        should_expand = (
            cpl_ratio <= 0.9 and
            leads_7d >= 1 and leads_7d <= 5 and
            share_of_parent_spend < 0.3  # Menos de 30% do spend da campanha
        )

        if not should_expand:
            return None

        entity_id = self._get_entity_id(features, "adset")

        return Recommendation(
            recommendation_type=RecommendationType.AUDIENCE_EXPANSION,
            priority=5,
            title="Expandir audiência - Performance boa, volume baixo",
            description=(
                f"Este conjunto de anúncios tem CPL de R$ {cpl_7d:.2f} ({cpl_ratio:.0%} da média), "
                f"mas está gerando apenas {leads_7d} leads/semana. "
                f"Representa apenas {share_of_parent_spend:.0%} do investimento da campanha. "
                f"Recomendamos expandir a audiência para capturar mais volume."
            ),
            suggested_action={
                "action": "expand_audience",
                "recommendations": [
                    "Aumentar faixa etária ou região",
                    "Testar interesses relacionados",
                    "Criar lookalike de maior percentual",
                    "Remover exclusões restritivas"
                ]
            },
            confidence_score=0.7,
            reasoning={
                "cpl_7d": cpl_7d,
                "cpl_ratio": cpl_ratio,
                "leads_7d": leads_7d,
                "share_of_parent_spend": share_of_parent_spend,
                "rule": "AUDIENCE_EXPANSION_GOOD_CPL_LOW_VOLUME"
            },
            entity_type="adset",
            entity_id=entity_id,
            config_id=features.config_id,
            expires_in_days=7
        )

    def _check_audience_narrowing(
        self,
        features: Union[CampaignFeatures, EntityFeatures]
    ) -> Optional[Recommendation]:
        """
        Regra: Reduzir audiência quando CPL alto mas CTR bom.
        Específico para adsets.

        Triggers:
        - CPL alto (acima da média)
        - CTR bom (indica interesse)
        - Baixa taxa de conversão pós-clique
        """
        is_active = getattr(features, 'is_active', True)
        if not is_active:
            return None

        cpl_7d = getattr(features, 'cpl_7d', 0)
        ctr_7d = getattr(features, 'ctr_7d', 0)
        conversion_rate_7d = getattr(features, 'conversion_rate_7d', 0)

        cpl_ratio = cpl_7d / self.avg_cpl if self.avg_cpl > 0 else 1.0
        ctr_ratio = ctr_7d / self.avg_ctr if self.avg_ctr > 0 else 1.0

        # CPL alto, CTR bom, conversão baixa
        should_narrow = (
            cpl_ratio >= 1.3 and
            ctr_ratio >= 1.0 and  # CTR igual ou melhor que média
            conversion_rate_7d < self.avg_conversion_rate * 0.7  # Conversão 30% abaixo
        )

        if not should_narrow:
            return None

        entity_id = self._get_entity_id(features, "adset")

        return Recommendation(
            recommendation_type=RecommendationType.AUDIENCE_NARROWING,
            priority=5,
            title="Refinar audiência - CPL alto com CTR bom",
            description=(
                f"Este conjunto de anúncios tem CPL de R$ {cpl_7d:.2f} ({cpl_ratio:.0%} da média), "
                f"mas o CTR de {ctr_7d:.2f}% está acima da média. "
                f"Isso indica interesse, mas baixa qualificação do público. "
                f"Recomendamos refinar a audiência para melhorar a conversão."
            ),
            suggested_action={
                "action": "narrow_audience",
                "recommendations": [
                    "Adicionar exclusões de públicos não-qualificados",
                    "Focar em interesses mais específicos",
                    "Reduzir faixa etária ou região",
                    "Usar lookalike de menor percentual"
                ]
            },
            confidence_score=0.7,
            reasoning={
                "cpl_7d": cpl_7d,
                "cpl_ratio": cpl_ratio,
                "ctr_7d": ctr_7d,
                "ctr_ratio": ctr_ratio,
                "conversion_rate_7d": conversion_rate_7d,
                "rule": "AUDIENCE_NARROWING_HIGH_CPL_GOOD_CTR"
            },
            entity_type="adset",
            entity_id=entity_id,
            config_id=features.config_id,
            expires_in_days=7
        )

    # ==================== AD-SPECIFIC RULES ====================

    def _check_creative_test(
        self,
        features: Union[CampaignFeatures, EntityFeatures]
    ) -> Optional[Recommendation]:
        """
        Regra: Sugerir teste A/B de criativos.
        Específico para ads.

        Triggers:
        - Performance mediana (não é claro se é bom ou ruim)
        - Volume de dados suficiente para teste
        """
        is_active = getattr(features, 'is_active', True)
        if not is_active:
            return None

        cpl_7d = getattr(features, 'cpl_7d', 0)
        impressions_7d = getattr(features, 'impressions_7d', 0)
        cpl_std_7d = getattr(features, 'cpl_std_7d', 0)

        cpl_ratio = cpl_7d / self.avg_cpl if self.avg_cpl > 0 else 1.0

        # Performance mediana com volume suficiente
        should_test = (
            0.9 <= cpl_ratio <= 1.2 and
            impressions_7d >= 1000 and
            cpl_std_7d > 0  # Alguma variação no CPL
        )

        if not should_test:
            return None

        entity_id = self._get_entity_id(features, "ad")

        return Recommendation(
            recommendation_type=RecommendationType.CREATIVE_TEST,
            priority=4,
            title="Testar variações de criativo",
            description=(
                f"Este anúncio tem performance mediana: CPL de R$ {cpl_7d:.2f} ({cpl_ratio:.0%} da média). "
                f"Com {impressions_7d:,} impressões, há dados suficientes para teste. "
                f"Recomendamos criar variações para identificar melhor abordagem."
            ),
            suggested_action={
                "action": "create_variations",
                "recommendations": [
                    "Criar 2-3 variações de headline",
                    "Testar diferentes imagens/vídeos",
                    "Variar call-to-action",
                    "Testar diferentes formatos"
                ]
            },
            confidence_score=0.65,
            reasoning={
                "cpl_7d": cpl_7d,
                "cpl_ratio": cpl_ratio,
                "impressions_7d": impressions_7d,
                "cpl_std_7d": cpl_std_7d,
                "rule": "CREATIVE_TEST_MEDIAN_PERFORMANCE"
            },
            entity_type="ad",
            entity_id=entity_id,
            config_id=features.config_id,
            expires_in_days=7
        )

    def _check_creative_winner(
        self,
        features: Union[CampaignFeatures, EntityFeatures],
        sibling_features: Optional[list] = None
    ) -> Optional[Recommendation]:
        """
        Regra: Identificar criativo vencedor para escalar.
        Específico para ads.

        Triggers:
        - Performance significativamente melhor que irmãos
        - Volume de dados suficiente para conclusão
        """
        is_active = getattr(features, 'is_active', True)
        if not is_active or not sibling_features:
            return None

        cpl_7d = getattr(features, 'cpl_7d', 0)
        leads_7d = getattr(features, 'leads_7d', 0)
        performance_vs_siblings = getattr(features, 'performance_vs_siblings', 0)

        # Precisa ser significativamente melhor que irmãos
        is_winner = (
            performance_vs_siblings >= 0.3 and  # 30% melhor que média dos irmãos
            leads_7d >= 3 and
            cpl_7d > 0
        )

        if not is_winner:
            return None

        entity_id = self._get_entity_id(features, "ad")

        return Recommendation(
            recommendation_type=RecommendationType.CREATIVE_WINNER,
            priority=7,
            title="Criativo vencedor identificado",
            description=(
                f"Este anúncio está {performance_vs_siblings * 100:.0f}% acima da média dos outros anúncios "
                f"no mesmo adset. Com CPL de R$ {cpl_7d:.2f} e {leads_7d} leads em 7 dias, "
                f"é um claro vencedor. Recomendamos aumentar o investimento neste criativo."
            ),
            suggested_action={
                "action": "scale_winner",
                "recommendations": [
                    "Aumentar budget do adset",
                    "Pausar anúncios de baixa performance",
                    "Criar variações baseadas neste criativo",
                    "Expandir para novos públicos"
                ],
                "expected_impact": {
                    "leads_increase": "20-40%",
                    "cpl_maintained": True,
                }
            },
            confidence_score=0.8,
            reasoning={
                "cpl_7d": cpl_7d,
                "leads_7d": leads_7d,
                "performance_vs_siblings": performance_vs_siblings,
                "sibling_count": len(sibling_features) if sibling_features else 0,
                "rule": "CREATIVE_WINNER_VS_SIBLINGS"
            },
            entity_type="ad",
            entity_id=entity_id,
            config_id=features.config_id,
            expires_in_days=7
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
