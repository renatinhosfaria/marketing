"""
Service for managing classification labels.
Supports heuristic labels (bootstrap) and real feedback labels.

This service breaks the circular bias in training by:
1. Prioritizing user feedback over heuristics
2. Tracking label sources for transparency
3. Supporting outcome-based labeling (future)
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from projects.ml.db.models import CampaignTier, MLClassificationFeedback
from projects.ml.services.feature_engineering import CampaignFeatures
from shared.config import settings
from shared.core.logging import get_logger

logger = get_logger(__name__)


class LabelSource(str, Enum):
    """Source of classification label."""

    HEURISTIC = "heuristic"  # Rule-based (bootstrap)
    USER_FEEDBACK = "user"  # User explicitly labeled
    OUTCOME_BASED = "outcome"  # Based on actual outcomes
    HYBRID = "hybrid"  # Combination of sources


class LabelService:
    """
    Service for generating and managing classification labels.

    Addresses the training bias problem by:
    - Preferring real feedback over heuristic labels
    - Tracking label sources for model transparency
    - Supporting future outcome-based labeling
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_training_labels(
        self,
        features_list: List[CampaignFeatures],
        avg_cpl: float,
        prefer_real_feedback: bool = True,
    ) -> Tuple[List[CampaignTier], List[LabelSource]]:
        """
        Get training labels, preferring real feedback over heuristics.

        Args:
            features_list: List of campaign features
            avg_cpl: Average CPL for reference
            prefer_real_feedback: If True, prefer user feedback over heuristics

        Returns:
            Tuple of (labels, sources) lists
        """
        labels = []
        sources = []

        for features in features_list:
            label, source = await self._get_label_for_campaign(
                features, avg_cpl, prefer_real_feedback
            )
            labels.append(label)
            sources.append(source)

        # Log label source distribution
        source_counts = {}
        for s in sources:
            source_counts[s.value] = source_counts.get(s.value, 0) + 1

        logger.info(
            "Training labels generated",
            total=len(labels),
            sources=source_counts,
            feedback_ratio=source_counts.get("user", 0) / len(labels)
            if len(labels) > 0
            else 0,
        )

        return labels, sources

    async def _get_label_for_campaign(
        self,
        features: CampaignFeatures,
        avg_cpl: float,
        prefer_real_feedback: bool,
    ) -> Tuple[CampaignTier, LabelSource]:
        """Get label for a single campaign."""

        if prefer_real_feedback:
            # 1. Check for user feedback
            user_label = await self._get_user_feedback_label(
                features.config_id, features.campaign_id
            )
            if user_label:
                return user_label, LabelSource.USER_FEEDBACK

            # 2. Check for outcome-based label (future implementation)
            outcome_label = await self._get_outcome_based_label(
                features.config_id, features.campaign_id, features
            )
            if outcome_label:
                return outcome_label, LabelSource.OUTCOME_BASED

        # 3. Fall back to heuristic
        heuristic_label = self._compute_heuristic_label(features, avg_cpl)
        return heuristic_label, LabelSource.HEURISTIC

    async def _get_user_feedback_label(
        self, config_id: int, campaign_id: str
    ) -> Optional[CampaignTier]:
        """
        Check if user has provided feedback on this campaign.

        Returns the most recent valid feedback label, or None if no feedback.
        """
        try:
            query = (
                select(MLClassificationFeedback)
                .where(
                    and_(
                        MLClassificationFeedback.config_id == config_id,
                        MLClassificationFeedback.entity_id == campaign_id,
                        MLClassificationFeedback.is_valid.is_(True),
                    )
                )
                .order_by(MLClassificationFeedback.created_at.desc())
                .limit(1)
            )

            result = await self.session.execute(query)
            feedback = result.scalar_one_or_none()

            if feedback:
                logger.debug(
                    "Using user feedback label",
                    config_id=config_id,
                    campaign_id=campaign_id,
                    correct_tier=feedback.correct_tier,
                )
                return CampaignTier(feedback.correct_tier)

            return None
        except Exception as e:
            # Table might not exist yet, fail gracefully
            logger.debug(
                "Could not fetch user feedback",
                config_id=config_id,
                campaign_id=campaign_id,
                error=str(e),
            )
            return None

    async def _get_outcome_based_label(
        self,
        config_id: int,
        campaign_id: str,
        features: CampaignFeatures,
    ) -> Optional[CampaignTier]:
        """
        Determine label based on actual outcomes over time.

        Key insight: Look at what happened AFTER a period, not just metrics.
        - Campaign was scaled up and performed well -> HIGH_PERFORMER
        - Campaign was paused due to poor results -> UNDERPERFORMER
        - etc.

        NOTE: This is a placeholder for future implementation.
        Requires tracking of actions taken and their outcomes.
        """
        # TODO: Implement outcome tracking
        # This would look at:
        # 1. Was the campaign scaled up after classification?
        # 2. Did it maintain/improve performance after scaling?
        # 3. Was it eventually paused due to poor performance?
        # 4. How long did it remain in each state?
        return None

    def _compute_heuristic_label(
        self, features: CampaignFeatures, avg_cpl: float
    ) -> CampaignTier:
        """
        Compute label using heuristics.

        These heuristics should match the business rules but are used
        ONLY when no real feedback is available.
        """
        cpl_ratio = features.cpl_7d / avg_cpl if avg_cpl > 0 else 1.0

        # Zero leads with significant spend -> definitely underperforming
        if features.leads_7d == 0 and features.spend_7d > 50:
            return CampaignTier.UNDERPERFORMER

        # High performer: low CPL + consistent leads
        if (
            cpl_ratio <= settings.threshold_cpl_low
            and features.days_with_leads_7d >= 4
        ):
            return CampaignTier.HIGH_PERFORMER

        # Moderate: acceptable CPL with some leads
        if cpl_ratio <= 1.0 and features.leads_7d >= 2:
            return CampaignTier.MODERATE

        # Low: above average CPL but not terrible
        if cpl_ratio <= settings.threshold_cpl_high:
            return CampaignTier.LOW

        # Underperformer: high CPL
        return CampaignTier.UNDERPERFORMER

    async def record_feedback(
        self,
        config_id: int,
        entity_id: str,
        entity_type: str,
        original_tier: CampaignTier,
        correct_tier: CampaignTier,
        user_id: int,
        classification_id: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> MLClassificationFeedback:
        """
        Record user feedback on a classification.

        Args:
            config_id: Configuration ID
            entity_id: Campaign/Adset/Ad ID
            entity_type: Type of entity
            original_tier: The original classification tier
            correct_tier: The correct tier according to user
            user_id: ID of the user providing feedback
            classification_id: Optional ID of the original classification
            reason: Optional reason for the correction

        Returns:
            Created feedback record
        """
        feedback = MLClassificationFeedback(
            config_id=config_id,
            entity_id=entity_id,
            entity_type=entity_type,
            original_tier=original_tier,
            correct_tier=correct_tier,
            user_id=user_id,
            original_classification_id=classification_id,
            feedback_reason=reason,
            is_valid=True,
            created_at=datetime.utcnow(),
        )

        self.session.add(feedback)
        await self.session.flush()

        logger.info(
            "Classification feedback recorded",
            config_id=config_id,
            entity_id=entity_id,
            original_tier=original_tier.value,
            correct_tier=correct_tier.value,
            user_id=user_id,
        )

        return feedback

    async def get_feedback_stats(self, config_id: int) -> dict:
        """
        Get statistics about classification feedback.

        Returns:
            Dict with feedback statistics
        """
        try:
            query = select(MLClassificationFeedback).where(
                MLClassificationFeedback.config_id == config_id
            )
            result = await self.session.execute(query)
            feedbacks = result.scalars().all()

            if not feedbacks:
                return {
                    "total_feedbacks": 0,
                    "by_original_tier": {},
                    "by_correct_tier": {},
                    "correction_rate": 0.0,
                }

            by_original = {}
            by_correct = {}
            corrections = 0

            for fb in feedbacks:
                # Count by original tier
                orig = fb.original_tier.value if fb.original_tier else "unknown"
                by_original[orig] = by_original.get(orig, 0) + 1

                # Count by correct tier
                corr = fb.correct_tier.value if fb.correct_tier else "unknown"
                by_correct[corr] = by_correct.get(corr, 0) + 1

                # Count actual corrections (different tiers)
                if fb.original_tier != fb.correct_tier:
                    corrections += 1

            return {
                "total_feedbacks": len(feedbacks),
                "by_original_tier": by_original,
                "by_correct_tier": by_correct,
                "corrections": corrections,
                "correction_rate": corrections / len(feedbacks) if feedbacks else 0.0,
            }
        except Exception as e:
            logger.warning("Could not fetch feedback stats", error=str(e))
            return {
                "total_feedbacks": 0,
                "error": str(e),
            }


async def get_label_service(session: AsyncSession) -> LabelService:
    """Factory for LabelService."""
    return LabelService(session)
