"""
Impact Analysis API endpoint.

Provides causal impact analysis for campaign changes.
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.session import get_db
from shared.core.logging import get_logger
from projects.ml.services.data_service import DataService
from projects.ml.algorithms.models.causal.impact_analyzer import (
    get_impact_analyzer,
    ImpactResult,
)

logger = get_logger(__name__)
router = APIRouter()


# ==================== REQUEST SCHEMAS ====================

class ImpactAnalysisRequest(BaseModel):
    """Request model for impact analysis."""

    config_id: int = Field(..., description="Facebook Ads configuration ID")
    entity_type: str = Field(
        ...,
        pattern="^(campaign|adset|ad)$",
        description="Entity type: campaign, adset, or ad"
    )
    entity_id: str = Field(..., description="Entity ID from Facebook")
    change_date: datetime = Field(..., description="Date when the change was made")
    change_type: str = Field(
        ...,
        pattern="^(budget_change|creative_change|audience_change|pause|reactivate)$",
        description="Type of change made"
    )
    window_before: int = Field(
        default=7,
        ge=3,
        le=30,
        description="Days before change to analyze"
    )
    window_after: int = Field(
        default=7,
        ge=3,
        le=30,
        description="Days after change to analyze"
    )


# ==================== RESPONSE SCHEMAS ====================

class MetricChange(BaseModel):
    """Individual metric change details."""

    metric: str = Field(..., description="Metric name (cpl, leads, ctr, spend)")
    pct_change: float = Field(..., description="Percentage change")
    is_significant: bool = Field(..., description="Whether change is statistically significant")
    confidence: float = Field(..., description="Statistical confidence level")
    effect_size: float = Field(..., description="Cohen's d effect size")


class ImpactAnalysisResponse(BaseModel):
    """Response model for impact analysis."""

    entity_type: str = Field(..., description="Entity type analyzed")
    entity_id: str = Field(..., description="Entity ID analyzed")
    change_date: datetime = Field(..., description="Date of the change")
    change_type: str = Field(..., description="Type of change analyzed")
    overall_impact: str = Field(
        ...,
        description="Overall impact assessment: positive, negative, neutral, or inconclusive"
    )
    recommendation: str = Field(..., description="Actionable recommendation based on analysis")
    metric_changes: list[MetricChange] = Field(
        ...,
        description="Detailed changes for each metric"
    )
    window_before: int = Field(..., description="Days before change analyzed")
    window_after: int = Field(..., description="Days after change analyzed")
    analyzed_at: datetime = Field(..., description="Timestamp of analysis")


# ==================== HELPER FUNCTIONS ====================

def convert_impact_result_to_response(result: ImpactResult) -> ImpactAnalysisResponse:
    """Convert ImpactResult dataclass to API response model."""
    metric_changes = []
    for metric in result.metric_changes.keys():
        metric_changes.append(
            MetricChange(
                metric=metric,
                pct_change=result.metric_changes[metric],
                is_significant=result.is_significant.get(metric, False),
                confidence=result.significance.get(metric, 0.0),
                effect_size=result.effect_sizes.get(metric, 0.0),
            )
        )

    return ImpactAnalysisResponse(
        entity_type=result.entity_type,
        entity_id=result.entity_id,
        change_date=result.change_date,
        change_type=result.change_type,
        overall_impact=result.overall_impact,
        recommendation=result.recommendation,
        metric_changes=metric_changes,
        window_before=result.window_before,
        window_after=result.window_after,
        analyzed_at=result.analyzed_at,
    )


# ==================== ENDPOINTS ====================

@router.post("/analyze", response_model=ImpactAnalysisResponse)
async def analyze_impact(
    request: ImpactAnalysisRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Analyze the causal impact of a change on entity performance.

    Compares performance metrics before and after a specified change date
    using statistical tests to determine significance.

    - **config_id**: Facebook Ads configuration ID
    - **entity_type**: Type of entity (campaign, adset, ad)
    - **entity_id**: ID of the entity to analyze
    - **change_date**: Date when the change was made
    - **change_type**: Type of change (budget_change, creative_change, audience_change, pause, reactivate)
    - **window_before**: Days before change to analyze (3-30, default 7)
    - **window_after**: Days after change to analyze (3-30, default 7)

    Returns statistical analysis including:
    - Overall impact assessment (positive, negative, neutral, inconclusive)
    - Per-metric changes with significance tests
    - Actionable recommendations
    """
    logger.info(
        "Impact analysis requested",
        config_id=request.config_id,
        entity_type=request.entity_type,
        entity_id=request.entity_id,
        change_type=request.change_type,
        change_date=request.change_date.isoformat(),
    )

    try:
        # Create services
        data_service = DataService(db)
        analyzer = get_impact_analyzer()

        # Perform analysis
        result = await analyzer.analyze_impact(
            entity_type=request.entity_type,
            entity_id=request.entity_id,
            change_date=request.change_date,
            change_type=request.change_type,
            data_service=data_service,
            config_id=request.config_id,
            window_before=request.window_before,
            window_after=request.window_after,
        )

        logger.info(
            "Impact analysis completed",
            entity_id=request.entity_id,
            overall_impact=result.overall_impact,
            metrics_analyzed=len(result.metric_changes),
        )

        return convert_impact_result_to_response(result)

    except ValueError as e:
        logger.warning(
            "Impact analysis failed with validation error",
            entity_id=request.entity_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(
            "Unexpected error during impact analysis",
            entity_id=request.entity_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during impact analysis",
        )
