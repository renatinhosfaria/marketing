# Multi-Level ML Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand ML services to support analysis at campaign, adset, and ad levels.

**Architecture:** Add `entity_type` column to classifications table, create generic `EntityFeatures` dataclass, extend services to accept entity_type parameter, and create separate Celery jobs for each level.

**Tech Stack:** Python 3.10, SQLAlchemy 2.0, Alembic, Celery, FastAPI, pandas

---

## Task 1: Update Database Models

**Files:**
- Modify: `projects/ml/db/models.py:64-73` (RecommendationType enum)
- Modify: `projects/ml/db/models.py:208-253` (MLCampaignClassification → MLClassification)
- Modify: `projects/ml/db/models.py:370-401` (MLFeature - add entity_type)

**Step 1: Add new recommendation types to enum**

```python
class RecommendationType(str, enum.Enum):
    """Tipos de recomendacoes."""
    # Generic (all levels)
    BUDGET_INCREASE = "BUDGET_INCREASE"
    BUDGET_DECREASE = "BUDGET_DECREASE"
    PAUSE = "PAUSE"
    SCALE_UP = "SCALE_UP"
    REACTIVATE = "REACTIVATE"

    # Adset specific
    AUDIENCE_REVIEW = "AUDIENCE_REVIEW"
    AUDIENCE_EXPANSION = "AUDIENCE_EXPANSION"
    AUDIENCE_NARROWING = "AUDIENCE_NARROWING"

    # Ad specific
    CREATIVE_REFRESH = "CREATIVE_REFRESH"
    CREATIVE_TEST = "CREATIVE_TEST"
    CREATIVE_WINNER = "CREATIVE_WINNER"

    # Legacy (deprecated, kept for compatibility)
    PAUSE_CAMPAIGN = "PAUSE_CAMPAIGN"
    OPTIMIZE_SCHEDULE = "OPTIMIZE_SCHEDULE"
```

**Step 2: Rename MLCampaignClassification to MLClassification and add entity_type**

```python
class MLClassification(Base):
    """
    Classificacao de entidades por tier de performance.
    Suporta campanhas, adsets e ads.
    """
    __tablename__ = "ml_classifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Entity identification
    entity_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="campaign"
    )  # campaign, adset, ad
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)
    parent_id: Mapped[Optional[str]] = mapped_column(String(100))  # campaign_id for adset, adset_id for ad

    # Classificacao atual
    tier: Mapped[CampaignTier] = mapped_column(
        Enum(CampaignTier), nullable=False
    )
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Snapshot de metricas usadas
    metrics_snapshot: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    feature_importances: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)

    # Historico de mudancas
    previous_tier: Mapped[Optional[CampaignTier]] = mapped_column(
        Enum(CampaignTier)
    )
    tier_change_direction: Mapped[Optional[str]] = mapped_column(
        String(20)
    )  # improved, declined, stable

    # Validade
    classified_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    valid_until: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Versao do modelo
    model_version: Mapped[Optional[str]] = mapped_column(String(50))

    __table_args__ = (
        Index(
            "ix_ml_classifications_entity",
            "config_id", "entity_type", "entity_id", "classified_at"
        ),
        Index("ix_ml_classifications_tier", "config_id", "tier"),
        Index("ix_ml_classifications_type", "config_id", "entity_type"),
        {"extend_existing": True},
    )


# Alias for backward compatibility
MLCampaignClassification = MLClassification
```

**Step 3: Add entity_type to MLFeature**

```python
class MLFeature(Base):
    """
    Features extraidas de entidades para uso em modelos ML.
    """
    __tablename__ = "ml_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Entity identification
    entity_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="campaign"
    )
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)
    parent_id: Mapped[Optional[str]] = mapped_column(String(100))

    # Configuracao
    window_days: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Features calculadas
    features: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    insufficient_data: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    __table_args__ = (
        Index(
            "ix_ml_features_entity",
            "config_id", "entity_type", "entity_id", "feature_date"
        ),
        Index("ix_ml_features_window", "config_id", "window_days"),
        {"extend_existing": True},
    )
```

**Step 4: Run tests to verify model changes compile**

Run: `cd /var/www/famachat-ml && python -c "from projects.ml.db.models import *; print('Models OK')"`
Expected: "Models OK"

**Step 5: Commit**

```bash
git add projects/ml/db/models.py
git commit -m "feat(ml): add entity_type support to classifications and features models"
```

---

## Task 2: Create Alembic Migration

**Files:**
- Create: `alembic/versions/2026_02_03_multi_level_ml.py`

**Step 1: Create migration file**

```python
"""Add multi-level ML support

Revision ID: multi_level_ml_001
Revises:
Create Date: 2026-02-03
"""
from alembic import op
import sqlalchemy as sa

revision = 'multi_level_ml_001'
down_revision = None  # Update with actual previous revision
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Rename ml_campaign_classifications to ml_classifications
    op.rename_table('ml_campaign_classifications', 'ml_classifications')

    # Add entity_type column to ml_classifications
    op.add_column(
        'ml_classifications',
        sa.Column('entity_type', sa.String(20), nullable=False, server_default='campaign')
    )

    # Add parent_id column to ml_classifications
    op.add_column(
        'ml_classifications',
        sa.Column('parent_id', sa.String(100), nullable=True)
    )

    # Rename campaign_id to entity_id in ml_classifications
    op.alter_column('ml_classifications', 'campaign_id', new_column_name='entity_id')

    # Add entity_type column to ml_features
    op.add_column(
        'ml_features',
        sa.Column('entity_type', sa.String(20), nullable=False, server_default='campaign')
    )

    # Add parent_id column to ml_features
    op.add_column(
        'ml_features',
        sa.Column('parent_id', sa.String(100), nullable=True)
    )

    # Rename campaign_id to entity_id in ml_features
    op.alter_column('ml_features', 'campaign_id', new_column_name='entity_id')

    # Create new indexes
    op.create_index(
        'ix_ml_classifications_entity',
        'ml_classifications',
        ['config_id', 'entity_type', 'entity_id', 'classified_at']
    )
    op.create_index(
        'ix_ml_classifications_type',
        'ml_classifications',
        ['config_id', 'entity_type']
    )
    op.create_index(
        'ix_ml_features_entity',
        'ml_features',
        ['config_id', 'entity_type', 'entity_id', 'feature_date']
    )

    # Drop old indexes (if they exist)
    op.drop_index('ix_ml_classifications_campaign', table_name='ml_classifications', if_exists=True)
    op.drop_index('ix_ml_features_campaign', table_name='ml_features', if_exists=True)


def downgrade() -> None:
    # Drop new indexes
    op.drop_index('ix_ml_features_entity', table_name='ml_features')
    op.drop_index('ix_ml_classifications_type', table_name='ml_classifications')
    op.drop_index('ix_ml_classifications_entity', table_name='ml_classifications')

    # Rename entity_id back to campaign_id
    op.alter_column('ml_features', 'entity_id', new_column_name='campaign_id')
    op.alter_column('ml_classifications', 'entity_id', new_column_name='campaign_id')

    # Drop new columns
    op.drop_column('ml_features', 'parent_id')
    op.drop_column('ml_features', 'entity_type')
    op.drop_column('ml_classifications', 'parent_id')
    op.drop_column('ml_classifications', 'entity_type')

    # Rename table back
    op.rename_table('ml_classifications', 'ml_campaign_classifications')

    # Recreate old indexes
    op.create_index(
        'ix_ml_classifications_campaign',
        'ml_campaign_classifications',
        ['config_id', 'campaign_id', 'classified_at']
    )
    op.create_index(
        'ix_ml_features_campaign',
        'ml_features',
        ['config_id', 'campaign_id', 'feature_date']
    )
```

**Step 2: Commit migration**

```bash
git add alembic/versions/
git commit -m "feat(ml): add migration for multi-level ML support"
```

---

## Task 3: Update InsightsRepository for Adsets and Ads

**Files:**
- Modify: `projects/ml/db/repositories/insights_repo.py`

**Step 1: Add get_active_adsets method**

```python
async def get_active_adsets(
    self, config_id: int, campaign_id: Optional[str] = None
) -> list[SistemaFacebookAdsAdsets]:
    """Obtem todos os adsets ativos de uma configuracao."""
    query = select(SistemaFacebookAdsAdsets).where(
        and_(
            SistemaFacebookAdsAdsets.config_id == config_id,
            SistemaFacebookAdsAdsets.status == "ACTIVE"
        )
    )
    if campaign_id:
        query = query.where(SistemaFacebookAdsAdsets.campaign_id == campaign_id)
    result = await self.session.execute(query)
    return list(result.scalars().all())

async def get_all_adsets(
    self, config_id: int, campaign_id: Optional[str] = None
) -> list[SistemaFacebookAdsAdsets]:
    """Obtem todos os adsets de uma configuracao."""
    query = select(SistemaFacebookAdsAdsets).where(
        SistemaFacebookAdsAdsets.config_id == config_id
    )
    if campaign_id:
        query = query.where(SistemaFacebookAdsAdsets.campaign_id == campaign_id)
    result = await self.session.execute(query)
    return list(result.scalars().all())
```

**Step 2: Add get_active_ads method**

```python
async def get_active_ads(
    self, config_id: int, adset_id: Optional[str] = None
) -> list[SistemaFacebookAdsAds]:
    """Obtem todos os ads ativos de uma configuracao."""
    query = select(SistemaFacebookAdsAds).where(
        and_(
            SistemaFacebookAdsAds.config_id == config_id,
            SistemaFacebookAdsAds.status == "ACTIVE"
        )
    )
    if adset_id:
        query = query.where(SistemaFacebookAdsAds.adset_id == adset_id)
    result = await self.session.execute(query)
    return list(result.scalars().all())

async def get_all_ads(
    self, config_id: int, adset_id: Optional[str] = None
) -> list[SistemaFacebookAdsAds]:
    """Obtem todos os ads de uma configuracao."""
    query = select(SistemaFacebookAdsAds).where(
        SistemaFacebookAdsAds.config_id == config_id
    )
    if adset_id:
        query = query.where(SistemaFacebookAdsAds.adset_id == adset_id)
    result = await self.session.execute(query)
    return list(result.scalars().all())
```

**Step 3: Add generic get_entities method**

```python
async def get_active_entities(
    self, config_id: int, entity_type: str, parent_id: Optional[str] = None
) -> list:
    """
    Obtem entidades ativas por tipo.

    Args:
        config_id: ID da configuracao
        entity_type: 'campaign', 'adset', ou 'ad'
        parent_id: ID do parent (campaign_id para adset, adset_id para ad)
    """
    if entity_type == "campaign":
        return await self.get_active_campaigns(config_id)
    elif entity_type == "adset":
        return await self.get_active_adsets(config_id, campaign_id=parent_id)
    elif entity_type == "ad":
        return await self.get_active_ads(config_id, adset_id=parent_id)
    else:
        raise ValueError(f"Unknown entity_type: {entity_type}")

async def get_all_entities(
    self, config_id: int, entity_type: str, parent_id: Optional[str] = None
) -> list:
    """Obtem todas as entidades por tipo."""
    if entity_type == "campaign":
        return await self.get_all_campaigns(config_id)
    elif entity_type == "adset":
        return await self.get_all_adsets(config_id, campaign_id=parent_id)
    elif entity_type == "ad":
        return await self.get_all_ads(config_id, adset_id=parent_id)
    else:
        raise ValueError(f"Unknown entity_type: {entity_type}")
```

**Step 4: Add aggregation methods for adsets and ads**

```python
async def get_aggregated_metrics_by_adset(
    self,
    config_id: int,
    start_date: datetime,
    end_date: datetime,
    campaign_id: Optional[str] = None,
) -> pd.DataFrame:
    """Obtem metricas agregadas por adset no periodo."""
    df = await self.get_insights_as_dataframe(
        config_id, start_date, end_date, "adset"
    )
    if df.empty:
        return df

    agg_df = df.groupby(["campaign_id", "adset_id"]).agg({
        "impressions": "sum",
        "reach": "sum",
        "clicks": "sum",
        "spend": "sum",
        "leads": "sum",
        "conversions": "sum",
    }).reset_index()

    agg_df["ctr"] = (agg_df["clicks"] / agg_df["impressions"] * 100).fillna(0)
    agg_df["cpc"] = (agg_df["spend"] / agg_df["clicks"]).fillna(0)
    agg_df["cpl"] = (agg_df["spend"] / agg_df["leads"]).replace([float('inf')], 0).fillna(0)
    agg_df["days_active"] = df.groupby("adset_id")["date"].nunique().reindex(agg_df["adset_id"]).values

    if campaign_id:
        agg_df = agg_df[agg_df["campaign_id"] == campaign_id]

    return agg_df

async def get_aggregated_metrics_by_ad(
    self,
    config_id: int,
    start_date: datetime,
    end_date: datetime,
    adset_id: Optional[str] = None,
) -> pd.DataFrame:
    """Obtem metricas agregadas por ad no periodo."""
    df = await self.get_insights_as_dataframe(
        config_id, start_date, end_date, "ad"
    )
    if df.empty:
        return df

    agg_df = df.groupby(["campaign_id", "adset_id", "ad_id"]).agg({
        "impressions": "sum",
        "reach": "sum",
        "clicks": "sum",
        "spend": "sum",
        "leads": "sum",
        "conversions": "sum",
    }).reset_index()

    agg_df["ctr"] = (agg_df["clicks"] / agg_df["impressions"] * 100).fillna(0)
    agg_df["cpc"] = (agg_df["spend"] / agg_df["clicks"]).fillna(0)
    agg_df["cpl"] = (agg_df["spend"] / agg_df["leads"]).replace([float('inf')], 0).fillna(0)
    agg_df["days_active"] = df.groupby("ad_id")["date"].nunique().reindex(agg_df["ad_id"]).values

    if adset_id:
        agg_df = agg_df[agg_df["adset_id"] == adset_id]

    return agg_df
```

**Step 5: Commit**

```bash
git add projects/ml/db/repositories/insights_repo.py
git commit -m "feat(ml): add adset and ad repository methods"
```

---

## Task 4: Update ML Repository for Generic Classifications

**Files:**
- Modify: `projects/ml/db/repositories/ml_repo.py`

**Step 1: Update classification methods to use entity_type**

Update `create_classification`, `get_classification`, `get_classifications` methods to accept `entity_type` and `entity_id` instead of `campaign_id`.

**Step 2: Update feature methods to use entity_type**

Update `create_feature`, `feature_exists`, `get_features` methods to accept `entity_type` and `entity_id`.

**Step 3: Commit**

```bash
git add projects/ml/db/repositories/ml_repo.py
git commit -m "feat(ml): update ml_repo for generic entity support"
```

---

## Task 5: Create EntityFeatures Dataclass

**Files:**
- Modify: `projects/ml/services/feature_engineering.py`

**Step 1: Create EntityFeatures dataclass**

```python
@dataclass
class EntityFeatures:
    """Features genericas para qualquer entidade."""
    entity_type: str  # "campaign", "adset", "ad"
    entity_id: str
    config_id: int
    parent_id: Optional[str]  # campaign_id para adset, adset_id para ad

    # Metricas basicas (7d)
    spend_7d: float
    impressions_7d: int
    clicks_7d: int
    leads_7d: int

    # Metricas calculadas
    cpl_7d: float
    ctr_7d: float
    cpc_7d: float
    conversion_rate_7d: float

    # Tendencias
    cpl_trend: float
    leads_trend: float
    spend_trend: float
    ctr_trend: float

    # Metricas de periodo mais longo
    cpl_14d: float
    leads_14d: int
    cpl_30d: float
    leads_30d: int
    avg_daily_spend_30d: float

    # Volatilidade
    cpl_std_7d: float
    leads_std_7d: float

    # Sazonalidade
    best_day_of_week: int
    worst_day_of_week: int

    # Frequencia e alcance
    frequency_7d: float
    reach_7d: int

    # Consistencia
    days_with_leads_7d: int
    days_active: int

    # Status
    is_active: bool
    has_budget: bool

    # Contexto hierarquico (novo)
    share_of_parent_spend: float = 0.0
    share_of_parent_leads: float = 0.0
    performance_vs_siblings: float = 0.0  # CPL comparado com irmaos

    # Timestamp
    computed_at: datetime = None

    def __post_init__(self):
        if self.computed_at is None:
            self.computed_at = datetime.now()


# Alias para compatibilidade
CampaignFeatures = EntityFeatures
```

**Step 2: Add compute_entity_features method**

```python
def compute_entity_features(
    self,
    daily_data: pd.DataFrame,
    entity_info: dict,
    entity_type: str,
    parent_metrics: Optional[dict] = None,
    sibling_metrics: Optional[pd.DataFrame] = None,
    reference_date: Optional[datetime] = None
) -> EntityFeatures:
    """
    Calcula features de uma entidade a partir dos dados diarios.

    Args:
        daily_data: DataFrame com dados diarios
        entity_info: Dict com informacoes da entidade
        entity_type: 'campaign', 'adset', ou 'ad'
        parent_metrics: Metricas do parent (para calcular share)
        sibling_metrics: DataFrame com metricas dos irmaos
        reference_date: Data de referencia
    """
    # Reutiliza logica existente de compute_campaign_features
    # Adiciona calculo de share_of_parent e performance_vs_siblings
```

**Step 3: Commit**

```bash
git add projects/ml/services/feature_engineering.py
git commit -m "feat(ml): add EntityFeatures with hierarchical context"
```

---

## Task 6: Update DataService for Multi-Level Support

**Files:**
- Modify: `projects/ml/services/data_service.py`

**Step 1: Add generic entity methods**

```python
async def get_entity_features(
    self,
    config_id: int,
    entity_type: str,
    entity_id: str,
    parent_id: Optional[str] = None,
    days: int = 30,
) -> Optional[EntityFeatures]:
    """Obtem features de qualquer entidade."""

async def get_all_entity_features(
    self,
    config_id: int,
    entity_type: str,
    parent_id: Optional[str] = None,
    active_only: bool = True,
) -> list[EntityFeatures]:
    """Obtem features de todas as entidades de um tipo."""

async def get_entity_daily_data(
    self,
    config_id: int,
    entity_type: str,
    entity_id: str,
    days: int = 30,
) -> pd.DataFrame:
    """Obtem dados diarios de uma entidade."""
```

**Step 2: Commit**

```bash
git add projects/ml/services/data_service.py
git commit -m "feat(ml): add multi-level data service methods"
```

---

## Task 7: Update AnomalyService for Multi-Level

**Files:**
- Modify: `projects/ml/services/anomaly_service.py`

**Step 1: Add entity_type parameter to detect_anomalies**

```python
async def detect_anomalies(
    self,
    config_id: int,
    entity_type: str = "campaign",
    entity_ids: Optional[list[str]] = None,
    days_to_analyze: int = 1,
    history_days: int = 30,
) -> AnomalyDetectionResult:
```

**Step 2: Update internal logic to handle different entity types**

**Step 3: Commit**

```bash
git add projects/ml/services/anomaly_service.py
git commit -m "feat(ml): add multi-level support to anomaly service"
```

---

## Task 8: Update ClassificationService for Multi-Level

**Files:**
- Modify: `projects/ml/services/classification_service.py`

**Step 1: Add entity_type parameter to classify methods**

```python
async def classify_entities(
    self,
    config_id: int,
    entity_type: str = "campaign",
    entity_ids: Optional[list[str]] = None,
    force_reclassify: bool = False,
) -> list[dict]:
```

**Step 2: Commit**

```bash
git add projects/ml/services/classification_service.py
git commit -m "feat(ml): add multi-level support to classification service"
```

---

## Task 9: Update RecommendationService and RuleEngine

**Files:**
- Modify: `projects/ml/services/recommendation_service.py`
- Modify: `projects/ml/algorithms/models/recommendation/rule_engine.py`

**Step 1: Add level-specific rules to RuleEngine**

```python
def _get_rules_for_entity_type(self, entity_type: str) -> list[Rule]:
    """Retorna regras aplicaveis ao tipo de entidade."""
    common_rules = [...]  # SCALE_UP, PAUSE, etc.

    if entity_type == "adset":
        return common_rules + [AUDIENCE_REVIEW, AUDIENCE_EXPANSION, AUDIENCE_NARROWING]
    elif entity_type == "ad":
        return common_rules + [CREATIVE_REFRESH, CREATIVE_TEST, CREATIVE_WINNER]
    return common_rules
```

**Step 2: Add entity_type to generate_recommendations**

```python
async def generate_recommendations(
    self,
    config_id: int,
    entity_type: str = "campaign",
    entity_ids: Optional[list[str]] = None,
    force_refresh: bool = False,
) -> list[dict]:
```

**Step 3: Commit**

```bash
git add projects/ml/services/recommendation_service.py projects/ml/algorithms/models/recommendation/rule_engine.py
git commit -m "feat(ml): add level-specific rules to recommendation engine"
```

---

## Task 10: Update API Endpoints

**Files:**
- Modify: `projects/ml/api/classifications.py`
- Modify: `projects/ml/api/anomalies.py`
- Modify: `projects/ml/api/recommendations.py`
- Modify: `projects/ml/api/forecasts.py`

**Step 1: Add entity_type parameter to classification endpoints**

**Step 2: Add entity_type to detect and generate endpoints**

**Step 3: Commit**

```bash
git add projects/ml/api/
git commit -m "feat(ml): add entity_type parameter to all ML endpoints"
```

---

## Task 11: Create Separate Celery Jobs

**Files:**
- Modify: `projects/ml/jobs/scheduled_tasks.py`

**Step 1: Create level-specific job functions**

```python
async def _run_classification_for_entity_type(
    config_id: int,
    entity_type: str,
    session_maker=None
) -> dict:
    """Classificacao para um tipo de entidade."""

async def _run_anomaly_detection_for_entity_type(
    config_id: int,
    entity_type: str,
    session_maker=None
) -> dict:
    """Deteccao de anomalias para um tipo de entidade."""

async def _run_recommendations_for_entity_type(
    config_id: int,
    entity_type: str,
    session_maker=None
) -> dict:
    """Recomendacoes para um tipo de entidade."""
```

**Step 2: Create Celery tasks for each level**

```python
@celery_app.task(name="projects.ml.jobs.scheduled_tasks.adset_classification")
def adset_classification():
    """Classificacao de adsets. Executado as 05:30."""

@celery_app.task(name="projects.ml.jobs.scheduled_tasks.ad_classification")
def ad_classification():
    """Classificacao de ads. Executado as 06:00."""

@celery_app.task(name="projects.ml.jobs.scheduled_tasks.adset_anomaly_detection")
def adset_anomaly_detection():
    """Anomalias em adsets. Executado a cada hora."""

@celery_app.task(name="projects.ml.jobs.scheduled_tasks.ad_anomaly_detection")
def ad_anomaly_detection():
    """Anomalias em ads. Executado a cada 2 horas."""
```

**Step 3: Commit**

```bash
git add projects/ml/jobs/scheduled_tasks.py
git commit -m "feat(ml): add separate jobs for adset and ad levels"
```

---

## Task 12: Update Celery Beat Schedule

**Files:**
- Modify: `app/celery.py` (or wherever beat schedule is defined)

**Step 1: Add new schedules**

```python
'adset-classification': {
    'task': 'projects.ml.jobs.scheduled_tasks.adset_classification',
    'schedule': crontab(hour=5, minute=30),
},
'ad-classification': {
    'task': 'projects.ml.jobs.scheduled_tasks.ad_classification',
    'schedule': crontab(hour=6, minute=0),
},
'adset-anomaly-detection': {
    'task': 'projects.ml.jobs.scheduled_tasks.adset_anomaly_detection',
    'schedule': crontab(minute=30),  # Every hour at :30
},
'ad-anomaly-detection': {
    'task': 'projects.ml.jobs.scheduled_tasks.ad_anomaly_detection',
    'schedule': crontab(minute=0, hour='*/2'),  # Every 2 hours
},
```

**Step 2: Commit**

```bash
git add app/celery.py
git commit -m "feat(ml): add celery beat schedule for multi-level jobs"
```

---

## Task 13: Update Documentation

**Files:**
- Modify: `docs/ml.md`

**Step 1: Update architecture section with multi-level support**

**Step 2: Document new recommendation types**

**Step 3: Document new jobs and schedules**

**Step 4: Commit**

```bash
git add docs/ml.md
git commit -m "docs(ml): update documentation for multi-level support"
```

---

## Final Verification

Run the following to verify the implementation:

```bash
# 1. Check models compile
python -c "from projects.ml.db.models import *; print('Models OK')"

# 2. Run migration (in Docker)
docker-compose exec api alembic upgrade head

# 3. Test with real data
docker-compose exec api python -c "
import asyncio
from shared.db.session import async_session_maker
from projects.ml.services.anomaly_service import AnomalyService

async def test():
    async with async_session_maker() as session:
        service = AnomalyService(session)
        result = await service.detect_anomalies(14, entity_type='campaign')
        print(f'Campaign anomalies: {result.anomalies_detected}')
        result = await service.detect_anomalies(14, entity_type='adset')
        print(f'Adset anomalies: {result.anomalies_detected}')

asyncio.run(test())
"
```
