"""
Impact Analyzer for causal inference on campaign changes.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Literal

import numpy as np
import pandas as pd
from scipy import stats

from shared.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ImpactResult:
    """Result of impact analysis."""
    entity_type: str
    entity_id: str
    change_date: datetime
    change_type: str

    # Metric changes (% change)
    metric_changes: dict[str, float]

    # Statistical significance
    significance: dict[str, float]
    is_significant: dict[str, bool]

    # Effect sizes (Cohen's d)
    effect_sizes: dict[str, float]

    # Verdict
    overall_impact: Literal['positive', 'negative', 'neutral', 'inconclusive']
    recommendation: str

    # Analysis metadata
    window_before: int
    window_after: int
    analyzed_at: datetime


class ImpactAnalyzer:
    """
    Analyzes the causal impact of changes on campaign performance.

    Uses before/after comparison with statistical testing.
    """

    # Primary metrics by change type
    PRIMARY_METRICS = {
        'budget_change': ['cpl', 'leads'],
        'creative_change': ['ctr', 'cpl'],
        'audience_change': ['cpl', 'leads', 'ctr'],
        'pause': ['spend'],
        'reactivate': ['leads', 'cpl'],
    }

    def __init__(
        self,
        significance_threshold: float = 0.05,
        min_effect_size: float = 0.2,
    ):
        self.significance_threshold = significance_threshold
        self.min_effect_size = min_effect_size

    async def analyze_impact(
        self,
        entity_type: str,
        entity_id: str,
        change_date: datetime,
        change_type: str,
        data_service,
        config_id: int,
        window_before: int = 7,
        window_after: int = 7,
    ) -> ImpactResult:
        """
        Analyze the impact of a change on entity performance.
        """
        # Get data
        start_date = change_date - timedelta(days=window_before)
        end_date = change_date + timedelta(days=window_after)

        df = await data_service.get_entity_daily_data(
            config_id=config_id,
            entity_type=entity_type,
            entity_id=entity_id,
            start_date=start_date,
            end_date=end_date,
        )

        if df.empty:
            raise ValueError(f"No data for entity {entity_id}")

        # Split into before and after
        df['date'] = pd.to_datetime(df['date'])
        before_df = df[df['date'] < change_date]
        after_df = df[df['date'] >= change_date]

        if len(before_df) < 3 or len(after_df) < 3:
            raise ValueError("Insufficient data for before/after comparison")

        # Analyze each metric
        metrics_to_analyze = ['cpl', 'leads', 'ctr', 'spend']
        metric_changes = {}
        significance = {}
        is_significant = {}
        effect_sizes = {}

        for metric in metrics_to_analyze:
            if metric not in df.columns:
                continue

            result = self._analyze_metric(
                before_df[metric].dropna(),
                after_df[metric].dropna(),
            )

            if result:
                metric_changes[metric] = result['pct_change']
                significance[metric] = result['confidence']
                is_significant[metric] = result['is_significant']
                effect_sizes[metric] = result['effect_size']

        # Determine overall impact
        overall_impact = self._determine_overall_impact(
            metric_changes, is_significant, effect_sizes, change_type
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            metric_changes, is_significant, change_type, overall_impact
        )

        return ImpactResult(
            entity_type=entity_type,
            entity_id=entity_id,
            change_date=change_date,
            change_type=change_type,
            metric_changes=metric_changes,
            significance=significance,
            is_significant=is_significant,
            effect_sizes=effect_sizes,
            overall_impact=overall_impact,
            recommendation=recommendation,
            window_before=window_before,
            window_after=window_after,
            analyzed_at=datetime.utcnow(),
        )

    def _analyze_metric(
        self,
        before: pd.Series,
        after: pd.Series,
    ) -> Optional[dict]:
        """Analyze a single metric's before/after change."""
        if len(before) < 2 or len(after) < 2:
            return None

        before_mean = before.mean()
        after_mean = after.mean()

        # Percentage change
        if before_mean != 0:
            pct_change = ((after_mean - before_mean) / before_mean) * 100
        else:
            pct_change = float('inf') if after_mean > 0 else 0

        # T-test with NaN handling
        t_stat, p_value = stats.ttest_ind(before, after)

        # Handle NaN p-value (occurs with constant arrays)
        if np.isnan(p_value):
            p_value = 1.0  # No significant difference if t-test fails

        confidence = round(1 - p_value, 3)
        is_significant = p_value < self.significance_threshold

        # Cohen's d effect size with division by zero protection
        denominator = len(before) + len(after) - 2
        if denominator <= 0:
            # Edge case: not enough degrees of freedom
            cohens_d = 0.0
        else:
            pooled_std = np.sqrt(
                ((len(before) - 1) * before.std() ** 2 +
                 (len(after) - 1) * after.std() ** 2) /
                denominator
            )

            if pooled_std > 0:
                cohens_d = (after_mean - before_mean) / pooled_std
            else:
                cohens_d = 0.0

        return {
            'pct_change': round(pct_change, 2),
            'confidence': confidence,
            'is_significant': is_significant,
            'effect_size': round(cohens_d, 3),
        }

    def _determine_overall_impact(
        self,
        metric_changes: dict,
        is_significant: dict,
        effect_sizes: dict,
        change_type: str,
    ) -> Literal['positive', 'negative', 'neutral', 'inconclusive']:
        """Determine overall impact based on key metrics."""
        key_metrics = self.PRIMARY_METRICS.get(change_type, ['cpl', 'leads'])

        positive_signals = 0
        negative_signals = 0

        for metric in key_metrics:
            if metric not in metric_changes:
                continue

            change = metric_changes[metric]
            significant = is_significant.get(metric, False)
            effect = abs(effect_sizes.get(metric, 0))

            # For CPL: negative change is positive
            # For leads/CTR: positive change is positive
            is_positive_change = (
                (metric == 'cpl' and change < 0) or
                (metric != 'cpl' and change > 0)
            )

            if significant and effect >= self.min_effect_size:
                if is_positive_change:
                    positive_signals += 1
                else:
                    negative_signals += 1

        if positive_signals > negative_signals and positive_signals > 0:
            return 'positive'
        elif negative_signals > positive_signals and negative_signals > 0:
            return 'negative'
        elif positive_signals == 0 and negative_signals == 0:
            return 'inconclusive'
        else:
            return 'neutral'

    def _generate_recommendation(
        self,
        metric_changes: dict,
        is_significant: dict,
        change_type: str,
        overall_impact: str,
    ) -> str:
        """Generate actionable recommendation."""
        recommendations = {
            ('positive', 'budget_change'): "Budget change was beneficial. Consider further scaling if metrics remain stable.",
            ('positive', 'creative_change'): "New creative is performing better. Continue with this direction.",
            ('positive', 'audience_change'): "Audience change improved performance. Consider similar expansions.",
            ('negative', 'budget_change'): "Budget change hurt performance. Consider reverting or adjusting.",
            ('negative', 'creative_change'): "New creative underperforms. Consider reverting or A/B testing alternatives.",
            ('negative', 'audience_change'): "Audience change degraded performance. Consider reverting.",
            ('inconclusive', None): "Not enough data to determine impact. Continue monitoring for 3-7 more days.",
            ('neutral', None): "Change had minimal impact. No action required.",
        }

        key = (overall_impact, change_type)
        if key in recommendations:
            return recommendations[key]

        # Fallback
        key = (overall_impact, None)
        return recommendations.get(key, "Continue monitoring.")


def get_impact_analyzer(
    significance_threshold: float = 0.05,
    min_effect_size: float = 0.2,
) -> ImpactAnalyzer:
    """Factory function for ImpactAnalyzer."""
    return ImpactAnalyzer(
        significance_threshold=significance_threshold,
        min_effect_size=min_effect_size,
    )
