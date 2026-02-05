"""
In-memory model cache with TTL for ML models.
Thread-safe implementation using locks for concurrent access.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import threading

from shared.config import settings
from shared.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CachedModel:
    """Cached model with metadata."""

    model: Any
    loaded_at: datetime
    model_path: str
    config_id: int
    model_type: str
    hits: int = 0


@dataclass
class CacheStats:
    """Statistics for the model cache."""

    cached_models: int
    total_hits: int
    total_misses: int
    models: list[dict] = field(default_factory=list)


class ModelCache:
    """
    Thread-safe in-memory cache for ML models.

    Features:
    - TTL-based expiration
    - Thread-safe operations
    - Statistics tracking
    - Singleton pattern
    """

    _instance: Optional["ModelCache"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelCache":
        if cls._instance is None:
            with cls._lock:
                # Double-check locking
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._cache: Dict[str, CachedModel] = {}
                    instance._cache_lock = threading.RLock()
                    instance._ttl = timedelta(
                        seconds=settings.model_cache_ttl_seconds
                    )
                    instance._total_hits = 0
                    instance._total_misses = 0
                    cls._instance = instance
        return cls._instance

    def _make_key(self, model_type: str, config_id: int) -> str:
        """Generate cache key from model type and config ID."""
        return f"{model_type}_{config_id}"

    def get(
        self, model_type: str, config_id: int
    ) -> Optional[Any]:
        """
        Get model from cache if valid.

        Args:
            model_type: Type of model (e.g., 'classifier', 'forecaster')
            config_id: Configuration ID

        Returns:
            Cached model or None if not found/expired
        """
        cache_key = self._make_key(model_type, config_id)

        with self._cache_lock:
            cached = self._cache.get(cache_key)

            if cached is None:
                self._total_misses += 1
                logger.debug(
                    "Cache miss",
                    key=cache_key,
                    model_type=model_type,
                    config_id=config_id
                )
                return None

            # Check TTL
            age = datetime.utcnow() - cached.loaded_at
            if age > self._ttl:
                del self._cache[cache_key]
                self._total_misses += 1
                logger.debug(
                    "Cache expired",
                    key=cache_key,
                    age_seconds=age.total_seconds()
                )
                return None

            # Cache hit
            cached.hits += 1
            self._total_hits += 1
            logger.debug(
                "Cache hit",
                key=cache_key,
                hits=cached.hits,
                age_seconds=age.total_seconds()
            )
            return cached.model

    def set(
        self,
        model_type: str,
        config_id: int,
        model: Any,
        model_path: str,
    ) -> None:
        """
        Store model in cache.

        Args:
            model_type: Type of model
            config_id: Configuration ID
            model: The model object to cache
            model_path: Path to the model file
        """
        cache_key = self._make_key(model_type, config_id)

        with self._cache_lock:
            self._cache[cache_key] = CachedModel(
                model=model,
                loaded_at=datetime.utcnow(),
                model_path=model_path,
                config_id=config_id,
                model_type=model_type,
            )
            logger.info(
                "Model cached",
                key=cache_key,
                model_type=model_type,
                config_id=config_id,
                model_path=model_path
            )

    def invalidate(self, model_type: str, config_id: int) -> bool:
        """
        Remove specific model from cache.

        Args:
            model_type: Type of model
            config_id: Configuration ID

        Returns:
            True if model was in cache and removed
        """
        cache_key = self._make_key(model_type, config_id)

        with self._cache_lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.info(
                    "Cache invalidated",
                    key=cache_key,
                    model_type=model_type,
                    config_id=config_id
                )
                return True
            return False

    def invalidate_by_type(self, model_type: str) -> int:
        """
        Remove all models of a specific type from cache.

        Args:
            model_type: Type of models to invalidate

        Returns:
            Number of models removed
        """
        with self._cache_lock:
            keys_to_remove = [
                k for k, v in self._cache.items()
                if v.model_type == model_type
            ]
            for key in keys_to_remove:
                del self._cache[key]

            if keys_to_remove:
                logger.info(
                    "Cache invalidated by type",
                    model_type=model_type,
                    count=len(keys_to_remove)
                )
            return len(keys_to_remove)

    def invalidate_by_config(self, config_id: int) -> int:
        """
        Remove all models for a specific config from cache.

        Args:
            config_id: Configuration ID

        Returns:
            Number of models removed
        """
        with self._cache_lock:
            keys_to_remove = [
                k for k, v in self._cache.items()
                if v.config_id == config_id
            ]
            for key in keys_to_remove:
                del self._cache[key]

            if keys_to_remove:
                logger.info(
                    "Cache invalidated by config",
                    config_id=config_id,
                    count=len(keys_to_remove)
                )
            return len(keys_to_remove)

    def clear(self) -> int:
        """
        Clear all cached models.

        Returns:
            Number of models cleared
        """
        with self._cache_lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info("Cache cleared", count=count)
            return count

    def stats(self) -> CacheStats:
        """
        Get cache statistics.

        Returns:
            CacheStats with current cache state
        """
        with self._cache_lock:
            now = datetime.utcnow()
            models = []

            for key, cached in self._cache.items():
                age = (now - cached.loaded_at).total_seconds()
                models.append({
                    "key": key,
                    "model_type": cached.model_type,
                    "config_id": cached.config_id,
                    "model_path": cached.model_path,
                    "hits": cached.hits,
                    "age_seconds": round(age, 2),
                    "ttl_remaining_seconds": round(
                        self._ttl.total_seconds() - age, 2
                    ),
                })

            return CacheStats(
                cached_models=len(self._cache),
                total_hits=self._total_hits,
                total_misses=self._total_misses,
                models=models,
            )

    def __len__(self) -> int:
        """Return number of cached models."""
        with self._cache_lock:
            return len(self._cache)


# Singleton instance
model_cache = ModelCache()


def get_model_cache() -> ModelCache:
    """
    Get the singleton model cache instance.

    Returns:
        ModelCache singleton instance
    """
    return model_cache
