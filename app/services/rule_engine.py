"""
Motor de regras para geração de recomendações.
Define regras de negócio para otimização de campanhas.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.core.logging import get_logger
from app.services.feature_engineering import CampaignFeatures

logger = get_logger(__name__)


class RuleType(str, Enum):
    """Tipos de regras."""
    BUDGET = "budget"
    PERFORMANCE = "performance"
    AUDIENCE = "audience"
    CREATIVE = "creative"
    SCHEDULE = "schedule"


class ActionType(str, Enum):
    """Tipos de ações recomendadas."""
    BUDGET_INCREASE = "BUDGET_INCREASE"
    BUDGET_DECREASE = "BUDGET_DECREASE"
    PAUSE_CAMPAIGN = "PAUSE_CAMPAIGN"
    SCALE_UP = "SCALE_UP"
    CREATIVE_REFRESH = "CREATIVE_REFRESH"
    AUDIENCE_REVIEW = "AUDIENCE_REVIEW"
    REACTIVATE = "REACTIVATE"
    OPTIMIZE_SCHEDULE = "OPTIMIZE_SCHEDULE"


@dataclass
class RuleResult:
    """Resultado da avaliação de uma regra."""
    triggered: bool
    rule_name: str
    rule_type: RuleType
    action_type: Optional[ActionType] = None
    priority: int = 5
    title: str = ""
    description: str = ""
    confidence: float = 0.5
    suggested_action: Optional[dict] = None
    reasoning: Optional[dict] = None


class RuleEngine:
    """
    Motor de regras para análise de campanhas.
    Avalia métricas e gera recomendações baseadas em regras de negócio.
    """
    
    def __init__(
        self,
        cpl_high_threshold: float = 150.0,
        cpl_low_threshold: float = 50.0,
        ctr_low_threshold: float = 0.5,
        frequency_high_threshold: float = 5.0,
        min_spend_for_analysis: float = 100.0,
    ):
        self.cpl_high_threshold = cpl_high_threshold
        self.cpl_low_threshold = cpl_low_threshold
        self.ctr_low_threshold = ctr_low_threshold
        self.frequency_high_threshold = frequency_high_threshold
        self.min_spend_for_analysis = min_spend_for_analysis
    
    def evaluate_campaign(
        self,
        features: CampaignFeatures,
        avg_cpl: float = 100.0,
    ) -> list[RuleResult]:
        """
        Avalia uma campanha e retorna recomendações.
        
        Args:
            features: Features calculadas da campanha
            avg_cpl: CPL médio da conta para comparação
            
        Returns:
            Lista de recomendações geradas
        """
        results = []
        
        # Verificar se há dados suficientes
        if features.total_spend < self.min_spend_for_analysis:
            return results
        
        # Regra 1: CPL muito alto - reduzir budget ou pausar
        cpl_rule = self._evaluate_cpl_rule(features, avg_cpl)
        if cpl_rule.triggered:
            results.append(cpl_rule)
        
        # Regra 2: CPL baixo e bom volume - escalar
        scale_rule = self._evaluate_scale_rule(features, avg_cpl)
        if scale_rule.triggered:
            results.append(scale_rule)
        
        # Regra 3: CTR muito baixo - revisar criativo
        ctr_rule = self._evaluate_ctr_rule(features)
        if ctr_rule.triggered:
            results.append(ctr_rule)
        
        # Regra 4: Frequência alta - fadiga de audiência
        frequency_rule = self._evaluate_frequency_rule(features)
        if frequency_rule.triggered:
            results.append(frequency_rule)
        
        # Regra 5: Tendência de piora - alertar
        trend_rule = self._evaluate_trend_rule(features)
        if trend_rule.triggered:
            results.append(trend_rule)
        
        logger.debug(
            "Regras avaliadas",
            campaign_id=features.campaign_id,
            rules_triggered=len(results)
        )
        
        return results
    
    def _evaluate_cpl_rule(
        self,
        features: CampaignFeatures,
        avg_cpl: float
    ) -> RuleResult:
        """Avalia regra de CPL alto."""
        if features.cpl <= 0:
            return RuleResult(triggered=False, rule_name="cpl_high", rule_type=RuleType.BUDGET)
        
        cpl_ratio = features.cpl / avg_cpl if avg_cpl > 0 else 1
        
        # CPL muito alto (>150% da média)
        if cpl_ratio > 1.5 or features.cpl > self.cpl_high_threshold:
            priority = 8 if cpl_ratio > 2 else 6
            
            if features.cpl > self.cpl_high_threshold * 1.5:
                # CPL extremamente alto - pausar
                return RuleResult(
                    triggered=True,
                    rule_name="cpl_very_high",
                    rule_type=RuleType.BUDGET,
                    action_type=ActionType.PAUSE_CAMPAIGN,
                    priority=9,
                    title="CPL Extremamente Alto",
                    description=(
                        f"CPL de R$ {features.cpl:.2f} está {cpl_ratio:.0%} acima da média. "
                        "Considere pausar a campanha para análise."
                    ),
                    confidence=0.85,
                    suggested_action={
                        "action": "pause",
                        "current_cpl": features.cpl,
                        "target_cpl": avg_cpl,
                    },
                    reasoning={
                        "cpl_ratio": cpl_ratio,
                        "threshold_exceeded": True,
                        "recommendation": "pause_for_review"
                    }
                )
            else:
                # CPL alto - reduzir budget
                return RuleResult(
                    triggered=True,
                    rule_name="cpl_high",
                    rule_type=RuleType.BUDGET,
                    action_type=ActionType.BUDGET_DECREASE,
                    priority=priority,
                    title="CPL Acima da Média",
                    description=(
                        f"CPL de R$ {features.cpl:.2f} está {(cpl_ratio - 1) * 100:.0f}% "
                        "acima da média da conta. Considere reduzir o orçamento."
                    ),
                    confidence=0.75,
                    suggested_action={
                        "action": "reduce_budget",
                        "reduction_percent": min(30, int((cpl_ratio - 1) * 50)),
                        "current_cpl": features.cpl,
                    },
                    reasoning={
                        "cpl_ratio": cpl_ratio,
                        "current_cpl": features.cpl,
                        "avg_cpl": avg_cpl,
                    }
                )
        
        return RuleResult(triggered=False, rule_name="cpl_high", rule_type=RuleType.BUDGET)
    
    def _evaluate_scale_rule(
        self,
        features: CampaignFeatures,
        avg_cpl: float
    ) -> RuleResult:
        """Avalia oportunidade de escalar campanha."""
        if features.cpl <= 0 or features.total_leads < 5:
            return RuleResult(triggered=False, rule_name="scale_up", rule_type=RuleType.BUDGET)
        
        cpl_ratio = features.cpl / avg_cpl if avg_cpl > 0 else 1
        
        # CPL baixo (< 70% da média) e bom volume
        if cpl_ratio < 0.7 and features.cpl < self.cpl_low_threshold:
            return RuleResult(
                triggered=True,
                rule_name="scale_up",
                rule_type=RuleType.BUDGET,
                action_type=ActionType.SCALE_UP,
                priority=7,
                title="Oportunidade de Escalar",
                description=(
                    f"Campanha com CPL de R$ {features.cpl:.2f} ({(1 - cpl_ratio) * 100:.0f}% "
                    "abaixo da média). Considere aumentar o orçamento para escalar resultados."
                ),
                confidence=0.8,
                suggested_action={
                    "action": "increase_budget",
                    "increase_percent": min(50, int((1 - cpl_ratio) * 100)),
                    "current_cpl": features.cpl,
                },
                reasoning={
                    "cpl_ratio": cpl_ratio,
                    "leads_volume": features.total_leads,
                    "trend": features.cpl_trend,
                }
            )
        
        return RuleResult(triggered=False, rule_name="scale_up", rule_type=RuleType.BUDGET)
    
    def _evaluate_ctr_rule(self, features: CampaignFeatures) -> RuleResult:
        """Avalia regra de CTR baixo."""
        if features.ctr < self.ctr_low_threshold and features.total_impressions > 1000:
            return RuleResult(
                triggered=True,
                rule_name="ctr_low",
                rule_type=RuleType.CREATIVE,
                action_type=ActionType.CREATIVE_REFRESH,
                priority=6,
                title="CTR Abaixo do Ideal",
                description=(
                    f"CTR de {features.ctr:.2f}% está abaixo do mínimo recomendado "
                    f"({self.ctr_low_threshold}%). Considere atualizar os criativos."
                ),
                confidence=0.7,
                suggested_action={
                    "action": "refresh_creative",
                    "current_ctr": features.ctr,
                    "target_ctr": self.ctr_low_threshold,
                },
                reasoning={
                    "ctr": features.ctr,
                    "threshold": self.ctr_low_threshold,
                    "impressions": features.total_impressions,
                }
            )
        
        return RuleResult(triggered=False, rule_name="ctr_low", rule_type=RuleType.CREATIVE)
    
    def _evaluate_frequency_rule(self, features: CampaignFeatures) -> RuleResult:
        """Avalia regra de frequência alta."""
        if features.avg_frequency > self.frequency_high_threshold:
            severity = "alta" if features.avg_frequency > 7 else "moderada"
            return RuleResult(
                triggered=True,
                rule_name="frequency_high",
                rule_type=RuleType.AUDIENCE,
                action_type=ActionType.AUDIENCE_REVIEW,
                priority=7 if features.avg_frequency > 7 else 5,
                title=f"Frequência {severity.title()}",
                description=(
                    f"Frequência de {features.avg_frequency:.1f} indica possível "
                    "saturação da audiência. Considere expandir ou renovar o público."
                ),
                confidence=0.75,
                suggested_action={
                    "action": "review_audience",
                    "current_frequency": features.avg_frequency,
                    "ideal_frequency": 3.0,
                },
                reasoning={
                    "frequency": features.avg_frequency,
                    "threshold": self.frequency_high_threshold,
                    "recommendation": "expand_audience"
                }
            )
        
        return RuleResult(triggered=False, rule_name="frequency_high", rule_type=RuleType.AUDIENCE)
    
    def _evaluate_trend_rule(self, features: CampaignFeatures) -> RuleResult:
        """Avalia tendência de piora."""
        # Tendência de CPL subindo e/ou leads caindo
        if features.cpl_trend > 0.15 and features.leads_trend < -0.1:
            return RuleResult(
                triggered=True,
                rule_name="negative_trend",
                rule_type=RuleType.PERFORMANCE,
                action_type=ActionType.BUDGET_DECREASE,
                priority=6,
                title="Tendência Negativa",
                description=(
                    "Campanha apresenta tendência de aumento de CPL "
                    f"({features.cpl_trend * 100:.0f}%) e queda de leads "
                    f"({features.leads_trend * 100:.0f}%). Monitorar de perto."
                ),
                confidence=0.65,
                suggested_action={
                    "action": "monitor_closely",
                    "cpl_trend": features.cpl_trend,
                    "leads_trend": features.leads_trend,
                },
                reasoning={
                    "cpl_trend": features.cpl_trend,
                    "leads_trend": features.leads_trend,
                    "alert_type": "performance_decline"
                }
            )
        
        return RuleResult(triggered=False, rule_name="negative_trend", rule_type=RuleType.PERFORMANCE)


# Instância global do motor de regras
rule_engine = RuleEngine()
