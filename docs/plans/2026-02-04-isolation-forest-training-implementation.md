# Isolation Forest Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Habilitar Isolation Forest por padrão e implementar treinamento automático diário por entidade.

**Architecture:** Modelos Isolation Forest treinados por entidade (campaign/adset/ad), persistidos em filesystem com joblib, carregados pelo AnomalyDetector durante detecção horária.

**Tech Stack:** sklearn.ensemble.IsolationForest, joblib, Celery Beat, asyncio

---

## Task 1: Adicionar configurações do Isolation Forest em settings.py

**Files:**
- Modify: `shared/infrastructure/config/settings.py:66-76`

**Step 1: Adicionar novas configurações**

No arquivo `shared/infrastructure/config/settings.py`, após a linha 76 (depois de `frequency_ideal`), adicionar:

```python
    # Isolation Forest configuration
    use_isolation_forest: bool = True  # Enable by default
    isolation_forest_min_samples: int = 50  # Minimum samples to train
    isolation_forest_contamination: float = 0.1  # Expected anomaly proportion
    isolation_forest_history_days: int = 90  # Days of history for training
```

**Step 2: Verificar sintaxe**

Run: `python -c "from shared.infrastructure.config.settings import settings; print(settings.use_isolation_forest)"`
Expected: `True`

**Step 3: Commit**

```bash
git add shared/infrastructure/config/settings.py
git commit -m "feat(ml): add Isolation Forest configuration settings"
```

---

## Task 2: Adicionar métodos de persistência no AnomalyDetector

**Files:**
- Modify: `projects/ml/algorithms/models/anomaly/anomaly_detector.py`

**Step 1: Adicionar imports necessários**

Após a linha 18 (`from scipy import stats`), adicionar:

```python
from pathlib import Path
import joblib
```

**Step 2: Modificar `__init__` para usar settings e adicionar cache**

Substituir o método `__init__` (linhas 114-134) por:

```python
    def __init__(
        self,
        z_threshold: Optional[float] = None,
        iqr_multiplier: Optional[float] = None,
        min_history_days: Optional[int] = None,
        use_isolation_forest: Optional[bool] = None,
    ):
        from shared.config import settings

        # Use settings as defaults if not provided
        self.z_threshold = z_threshold if z_threshold is not None else settings.anomaly_z_threshold
        self.iqr_multiplier = iqr_multiplier if iqr_multiplier is not None else settings.anomaly_iqr_multiplier
        self.min_history_days = min_history_days if min_history_days is not None else settings.anomaly_min_history_days

        # Isolation Forest configuration - use settings if not provided
        if use_isolation_forest is None:
            use_isolation_forest = settings.use_isolation_forest
        self.use_isolation_forest = use_isolation_forest and ISOLATION_FOREST_AVAILABLE
        self.isolation_forest_model: Optional[object] = None
        self.isolation_forest_features: list[str] = []

        # Model cache for loaded models (key: "config_id:entity_type:entity_id")
        self._models_cache: dict[str, tuple[object, list[str]]] = {}

        if use_isolation_forest and not ISOLATION_FOREST_AVAILABLE:
            logger.warning("Isolation Forest requested but sklearn not available")
```

**Step 3: Adicionar método `get_model_path`**

Após o método `__init__`, adicionar:

```python
    def get_model_path(self, config_id: int, entity_type: str, entity_id: str) -> Path:
        """Get filesystem path for a model."""
        from shared.config import settings
        base_path = Path(settings.models_storage_path) / "anomaly_detector" / f"config_{config_id}"
        return base_path / f"{entity_type}_{entity_id}.joblib"
```

**Step 4: Adicionar método `save_model`**

Após `get_model_path`, adicionar:

```python
    def save_model(self, config_id: int, entity_type: str, entity_id: str) -> bool:
        """
        Save trained Isolation Forest model to filesystem.

        Args:
            config_id: Facebook Ads config ID
            entity_type: 'campaign', 'adset', or 'ad'
            entity_id: Entity ID

        Returns:
            True if saved successfully
        """
        if self.isolation_forest_model is None:
            logger.warning("No model to save")
            return False

        model_path = self.get_model_path(config_id, entity_type, entity_id)

        try:
            # Create directory if not exists
            model_path.parent.mkdir(parents=True, exist_ok=True)

            # Save model and features together
            model_data = {
                'model': self.isolation_forest_model,
                'features': self.isolation_forest_features,
                'trained_at': datetime.utcnow().isoformat(),
                'entity_type': entity_type,
                'entity_id': entity_id,
            }
            joblib.dump(model_data, model_path)

            logger.info(
                "Model saved",
                config_id=config_id,
                entity_type=entity_type,
                entity_id=entity_id,
                path=str(model_path)
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to save model",
                error=str(e),
                path=str(model_path)
            )
            return False
```

**Step 5: Adicionar método `load_model`**

Após `save_model`, adicionar:

```python
    def load_model(self, config_id: int, entity_type: str, entity_id: str) -> bool:
        """
        Load Isolation Forest model from filesystem.
        Uses in-memory cache to avoid repeated disk reads.

        Args:
            config_id: Facebook Ads config ID
            entity_type: 'campaign', 'adset', or 'ad'
            entity_id: Entity ID

        Returns:
            True if loaded successfully
        """
        cache_key = f"{config_id}:{entity_type}:{entity_id}"

        # Check cache first
        if cache_key in self._models_cache:
            self.isolation_forest_model, self.isolation_forest_features = self._models_cache[cache_key]
            return True

        model_path = self.get_model_path(config_id, entity_type, entity_id)

        if not model_path.exists():
            return False

        try:
            model_data = joblib.load(model_path)
            self.isolation_forest_model = model_data['model']
            self.isolation_forest_features = model_data['features']

            # Cache for future use
            self._models_cache[cache_key] = (
                self.isolation_forest_model,
                self.isolation_forest_features
            )

            logger.debug(
                "Model loaded",
                config_id=config_id,
                entity_type=entity_type,
                entity_id=entity_id
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to load model",
                error=str(e),
                path=str(model_path)
            )
            return False
```

**Step 6: Verificar sintaxe**

Run: `python -c "from projects.ml.algorithms.models.anomaly.anomaly_detector import AnomalyDetector; d = AnomalyDetector(); print('OK')"`
Expected: `OK`

**Step 7: Commit**

```bash
git add projects/ml/algorithms/models/anomaly/anomaly_detector.py
git commit -m "feat(ml): add model persistence methods to AnomalyDetector"
```

---

## Task 3: Modificar detect_anomalies para carregar modelo

**Files:**
- Modify: `projects/ml/algorithms/models/anomaly/anomaly_detector.py`

**Step 1: Modificar assinatura de detect_anomalies**

Substituir a assinatura do método `detect_anomalies` (linhas 253-259) por:

```python
    def detect_anomalies(
        self,
        df: pd.DataFrame,
        entity_type: str,
        entity_id: str,
        config_id: Optional[int] = None,
        analysis_date: Optional[datetime] = None,
    ) -> list[DetectedAnomaly]:
        """
        Detecta anomalias em um DataFrame de métricas diárias.

        Args:
            df: DataFrame com colunas: date, spend, impressions, clicks, leads, etc.
            entity_type: Tipo da entidade (campaign, adset, ad)
            entity_id: ID da entidade
            config_id: ID da config (necessário para carregar modelo IF)
            analysis_date: Data para análise (padrão: última data no df)

        Returns:
            Lista de anomalias detectadas
        """
```

**Step 2: Modificar lógica de detecção multivariada**

Substituir as linhas 394-402 (seção 7 do detect_anomalies) por:

```python
        # 7. Detecção multivariada com Isolation Forest (se habilitado e modelo disponível)
        if self.use_isolation_forest and config_id is not None:
            # Try to load model for this entity
            if self.load_model(config_id, entity_type, entity_id):
                multivariate_anomalies = self._detect_multivariate_anomalies(
                    df_analysis,
                    entity_type,
                    entity_id,
                    analysis_date
                )
                anomalies.extend(multivariate_anomalies)
```

**Step 3: Verificar sintaxe**

Run: `python -c "from projects.ml.algorithms.models.anomaly.anomaly_detector import AnomalyDetector; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add projects/ml/algorithms/models/anomaly/anomaly_detector.py
git commit -m "feat(ml): integrate model loading in detect_anomalies"
```

---

## Task 4: Atualizar AnomalyService para passar config_id

**Files:**
- Modify: `projects/ml/services/anomaly_service.py`

**Step 1: Atualizar import para usar get_anomaly_detector**

Substituir as linhas 16-20 por:

```python
from projects.ml.algorithms.models.anomaly.anomaly_detector import (
    get_anomaly_detector,
    DetectedAnomaly,
    SeverityLevel,
)
```

**Step 2: Modificar chamada do detector em detect_anomalies**

Substituir as linhas 152-157 (chamada do anomaly_detector.detect_anomalies) por:

```python
                # Detectar anomalias
                detector = get_anomaly_detector()
                anomalies = detector.detect_anomalies(
                    df=df,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    config_id=config_id,
                )
```

**Step 3: Verificar sintaxe**

Run: `python -c "from projects.ml.services.anomaly_service import AnomalyService; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add projects/ml/services/anomaly_service.py
git commit -m "feat(ml): pass config_id to AnomalyDetector for IF model loading"
```

---

## Task 5: Implementar task de treinamento

**Files:**
- Modify: `projects/ml/jobs/training_tasks.py`

**Step 1: Adicionar imports necessários**

Após a linha 14 (`logger = get_logger(__name__)`), adicionar:

```python
import json
from pathlib import Path
```

**Step 2: Implementar função auxiliar de treinamento**

Após a task `train_anomaly_detector` (linha 147), adicionar:

```python

async def _train_isolation_forest_for_entity(
    config_id: int,
    entity_type: str,
    entity_id: str,
    session_maker,
) -> dict:
    """
    Train Isolation Forest for a single entity.

    Returns:
        Dict with training result
    """
    from projects.ml.services.data_service import DataService
    from projects.ml.algorithms.models.anomaly.anomaly_detector import AnomalyDetector
    from shared.config import settings

    async with session_maker() as session:
        data_service = DataService(session)

        # Get historical data
        df = await data_service.get_entity_daily_data(
            config_id=config_id,
            entity_type=entity_type,
            entity_id=entity_id,
            days=settings.isolation_forest_history_days,
        )

        if df.empty or len(df) < settings.isolation_forest_min_samples:
            return {
                "status": "skipped",
                "reason": "insufficient_data",
                "samples": len(df) if not df.empty else 0,
            }

        # Create detector and train
        detector = AnomalyDetector(use_isolation_forest=True)
        success = detector.train_isolation_forest(
            training_data=df,
            contamination=settings.isolation_forest_contamination,
        )

        if not success:
            return {
                "status": "failed",
                "reason": "training_failed",
            }

        # Save model
        saved = detector.save_model(config_id, entity_type, entity_id)

        return {
            "status": "success" if saved else "save_failed",
            "samples": len(df),
            "features": detector.isolation_forest_features,
        }


async def _train_isolation_forest_for_config(config_id: int, session_maker) -> dict:
    """
    Train Isolation Forest models for all active entities in a config.
    """
    from projects.ml.db.repositories.insights_repo import InsightsRepository
    from shared.config import settings
    import json
    from pathlib import Path
    from datetime import datetime

    results = {
        "config_id": config_id,
        "campaign": {"trained": 0, "skipped": 0, "failed": 0},
        "adset": {"trained": 0, "skipped": 0, "failed": 0},
        "ad": {"trained": 0, "skipped": 0, "failed": 0},
    }

    start_time = datetime.utcnow()

    async with session_maker() as session:
        insights_repo = InsightsRepository(session)

        for entity_type in ["campaign", "adset", "ad"]:
            entities = await insights_repo.get_active_entities(
                config_id=config_id,
                entity_type=entity_type,
            )

            logger.info(
                f"Training IF for {entity_type}s",
                config_id=config_id,
                count=len(entities),
            )

            for entity in entities:
                # Get entity ID based on type
                if entity_type == "campaign":
                    entity_id = entity.campaign_id
                elif entity_type == "adset":
                    entity_id = entity.adset_id
                else:
                    entity_id = entity.ad_id

                result = await _train_isolation_forest_for_entity(
                    config_id=config_id,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    session_maker=session_maker,
                )

                if result["status"] == "success":
                    results[entity_type]["trained"] += 1
                elif result["status"] == "skipped":
                    results[entity_type]["skipped"] += 1
                else:
                    results[entity_type]["failed"] += 1

    # Save metadata
    training_duration = (datetime.utcnow() - start_time).total_seconds()
    metadata = {
        "config_id": config_id,
        "last_training": datetime.utcnow().isoformat(),
        "models_count": {
            "campaign": results["campaign"]["trained"],
            "adset": results["adset"]["trained"],
            "ad": results["ad"]["trained"],
        },
        "training_duration_seconds": training_duration,
    }

    metadata_path = Path(settings.models_storage_path) / "anomaly_detector" / f"config_{config_id}" / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    results["training_duration_seconds"] = training_duration
    return results
```

**Step 3: Implementar a task principal**

Substituir a task `train_anomaly_detector` (linhas 139-147) por:

```python
@celery_app.task(
    name="projects.ml.jobs.training_tasks.train_anomaly_detector",
    max_retries=2,
)
def train_anomaly_detector(config_id: int):
    """Treina detector de anomalias Isolation Forest para uma config."""
    import asyncio
    from shared.db.session import create_isolated_async_session_maker

    logger.info("Iniciando treinamento anomaly detector", config_id=config_id)

    isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                _train_isolation_forest_for_config(config_id, isolated_session_maker)
            )
            loop.run_until_complete(isolated_engine.dispose())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        logger.info(
            "Treinamento IF concluído",
            config_id=config_id,
            campaigns_trained=result["campaign"]["trained"],
            adsets_trained=result["adset"]["trained"],
            ads_trained=result["ad"]["trained"],
        )

        return result

    except Exception as e:
        logger.error("Erro no treinamento IF", config_id=config_id, error=str(e))
        raise


@celery_app.task(
    name="projects.ml.jobs.training_tasks.train_anomaly_detectors_all",
    max_retries=1,
)
def train_anomaly_detectors_all():
    """
    Treina Isolation Forest para todas as configs ativas.
    Executado diariamente às 04:00.
    """
    from sqlalchemy.orm import sessionmaker
    from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig

    logger.info("Iniciando treinamento de todos os Isolation Forest")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        results = []
        for config in configs:
            logger.info(
                "Treinando IF para config",
                config_id=config.id,
                name=config.name,
            )
            result = train_anomaly_detector.delay(config.id)
            results.append({
                "config_id": config.id,
                "task_id": result.id,
            })

        logger.info(
            "Tasks de treinamento IF disparadas",
            configs_count=len(configs),
        )

        return {
            "status": "dispatched",
            "configs_count": len(configs),
            "tasks": results,
        }

    finally:
        session.close()
```

**Step 4: Verificar sintaxe**

Run: `python -c "from projects.ml.jobs.training_tasks import train_anomaly_detectors_all; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add projects/ml/jobs/training_tasks.py
git commit -m "feat(ml): implement Isolation Forest training tasks"
```

---

## Task 6: Adicionar schedule no Celery Beat

**Files:**
- Modify: `app/celery.py`

**Step 1: Adicionar schedule de treinamento**

Após a linha 96 (depois do `hourly-campaign-anomaly-detection`), adicionar:

```python
        # Treinar Isolation Forest diariamente às 04:00
        "daily-anomaly-detector-training": {
            "task": "projects.ml.jobs.training_tasks.train_anomaly_detectors_all",
            "schedule": crontab(hour=4, minute=0),
            "options": {"queue": "training"},
        },
```

**Step 2: Verificar sintaxe**

Run: `python -c "from app.celery import celery_app; print(len(celery_app.conf.beat_schedule))"`
Expected: Número maior que antes (ex: 17 ou mais)

**Step 3: Commit**

```bash
git add app/celery.py
git commit -m "feat(ml): add Isolation Forest training to Celery Beat schedule"
```

---

## Task 7: Atualizar factory function

**Files:**
- Modify: `projects/ml/algorithms/models/anomaly/anomaly_detector.py`

**Step 1: Atualizar get_anomaly_detector**

Substituir a função `get_anomaly_detector` (linhas 645-666) por:

```python
def get_anomaly_detector(
    z_threshold: Optional[float] = None,
    iqr_multiplier: Optional[float] = None,
    min_history_days: Optional[int] = None,
    use_isolation_forest: Optional[bool] = None,
) -> AnomalyDetector:
    """
    Factory function to create AnomalyDetector instances.
    Recommended for use in async/multi-threaded environments.

    Args:
        z_threshold: Custom Z-score threshold (defaults to settings)
        iqr_multiplier: Custom IQR multiplier (defaults to settings)
        min_history_days: Custom minimum history days (defaults to settings)
        use_isolation_forest: Enable Isolation Forest (defaults to settings)

    Returns:
        New AnomalyDetector instance
    """
    return AnomalyDetector(
        z_threshold=z_threshold,
        iqr_multiplier=iqr_multiplier,
        min_history_days=min_history_days,
        use_isolation_forest=use_isolation_forest,
    )
```

**Step 2: Verificar sintaxe**

Run: `python -c "from projects.ml.algorithms.models.anomaly.anomaly_detector import get_anomaly_detector; d = get_anomaly_detector(); print(d.use_isolation_forest)"`
Expected: `True`

**Step 3: Commit**

```bash
git add projects/ml/algorithms/models/anomaly/anomaly_detector.py
git commit -m "feat(ml): update factory function with Isolation Forest param"
```

---

## Task 8: Teste de integração manual

**Step 1: Verificar imports**

Run: `python -c "from projects.ml.jobs.training_tasks import train_anomaly_detectors_all, train_anomaly_detector; from projects.ml.services.anomaly_service import AnomalyService; from projects.ml.algorithms.models.anomaly.anomaly_detector import get_anomaly_detector; print('All imports OK')"`
Expected: `All imports OK`

**Step 2: Verificar settings**

Run: `python -c "from shared.config import settings; print(f'IF enabled: {settings.use_isolation_forest}'); print(f'Min samples: {settings.isolation_forest_min_samples}'); print(f'Contamination: {settings.isolation_forest_contamination}'); print(f'History days: {settings.isolation_forest_history_days}')"`
Expected:
```
IF enabled: True
Min samples: 50
Contamination: 0.1
History days: 90
```

**Step 3: Verificar Celery Beat schedule**

Run: `python -c "from app.celery import celery_app; schedule = celery_app.conf.beat_schedule; print('daily-anomaly-detector-training' in schedule)"`
Expected: `True`

**Step 4: Commit final**

```bash
git add -A
git commit -m "feat(ml): complete Isolation Forest training implementation (Phase 6)"
```

---

## Summary

Total tasks: 8
Files modified: 4
- `shared/infrastructure/config/settings.py`
- `projects/ml/algorithms/models/anomaly/anomaly_detector.py`
- `projects/ml/services/anomaly_service.py`
- `projects/ml/jobs/training_tasks.py`
- `app/celery.py`

New capabilities:
1. Isolation Forest enabled by default
2. Models trained per entity (campaign/adset/ad)
3. Models persisted in filesystem with joblib
4. Daily training at 04:00 via Celery Beat
5. Models loaded on-demand during anomaly detection
