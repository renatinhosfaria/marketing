"""
Serviço de classificação de campanhas.
Orquestra a classificação, armazenamento e gestão de tiers.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.ml_repo import MLRepository
from app.db.models.ml_models import (
    MLCampaignClassification,
    MLTrainedModel,
    CampaignTier,
    ModelType,
    ModelStatus,
)
from app.services.data_service import DataService
from app.services.feature_engineering import CampaignFeatures
from app.ml.models.classification.campaign_classifier import (
    CampaignClassifier,
    ClassificationResult,
    create_training_labels,
)
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ClassificationService:
    """
    Serviço para gerenciar classificação de campanhas.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.ml_repo = MLRepository(session)
        self.data_service = DataService(session)
        self.classifier = CampaignClassifier()
    
    async def classify_campaigns(
        self,
        config_id: int,
        campaign_ids: Optional[list[str]] = None,
        force_reclassify: bool = False
    ) -> list[dict]:
        """
        Classifica campanhas de uma configuração.
        
        Args:
            config_id: ID da configuração FB Ads
            campaign_ids: IDs específicos ou None para todas ativas
            force_reclassify: Se True, reclassifica mesmo se houver classificação válida
            
        Returns:
            Lista de classificações
        """
        logger.info(
            "Iniciando classificação de campanhas",
            config_id=config_id,
            campaign_ids=campaign_ids,
            force_reclassify=force_reclassify
        )
        
        # Verificar disponibilidade de dados (temporariamente desabilitado para debug)
        # data_check = await self.data_service.check_data_availability(config_id, min_days=3)
        # if not data_check['has_enough_data']:
        #     logger.warning(
        #         "Dados insuficientes para classificação",
        #         config_id=config_id,
        #         days_available=data_check['days_with_data']
        #     )
        #     return []
        logger.info("Verificação de dados desabilitada temporariamente", config_id=config_id)
        
        # Obter métricas de referência
        avg_metrics = await self.data_service.get_aggregated_metrics(config_id, days=14)
        avg_cpl = avg_metrics.get('avg_cpl', 50.0)
        avg_ctr = avg_metrics.get('avg_ctr', 1.0)
        
        # Tentar carregar modelo treinado
        await self._load_active_model(config_id)
        
        # Obter features das campanhas
        if campaign_ids:
            features_list = []
            for cid in campaign_ids:
                features = await self.data_service.get_campaign_features(config_id, cid)
                if features:
                    features_list.append(features)
        else:
            features_list = await self.data_service.get_all_campaign_features(
                config_id, active_only=True
            )
        
        if not features_list:
            logger.info("Nenhuma campanha com features disponíveis", config_id=config_id)
            return []
        
        # Classificar cada campanha
        classifications = []
        
        for features in features_list:
            # Verificar classificação existente (se não forçar reclassificação)
            if not force_reclassify:
                existing = await self.ml_repo.get_latest_classification(
                    config_id, features.campaign_id
                )
                if existing and existing.valid_until > datetime.utcnow():
                    classifications.append(self._to_response_dict(existing))
                    continue
            
            # Classificar
            result = self.classifier.classify(features, avg_cpl, avg_ctr)
            model_version = (
                self.classifier.model_version
                if self.classifier.is_fitted
                else "heuristic_v1"
            )
            
            # Obter tier anterior para tracking de mudanças
            previous = await self.ml_repo.get_latest_classification(
                config_id, features.campaign_id
            )
            previous_tier = previous.tier if previous else None
            
            # Salvar classificação
            saved = await self.ml_repo.create_classification(
                config_id=config_id,
                campaign_id=features.campaign_id,
                tier=result.tier,
                confidence_score=result.confidence_score,
                metrics_snapshot=result.metrics_snapshot,
                feature_importances=result.feature_importances,
                previous_tier=previous_tier,
                model_version=model_version,
            )
            
            classifications.append(self._to_response_dict(saved))
        
        logger.info(
            "Classificação concluída",
            config_id=config_id,
            total_classified=len(classifications)
        )
        
        return classifications
    
    async def get_campaign_classification(
        self,
        config_id: int,
        campaign_id: str
    ) -> Optional[dict]:
        """Obtém classificação mais recente de uma campanha."""
        classification = await self.ml_repo.get_latest_classification(
            config_id, campaign_id
        )
        
        if not classification:
            return None
        
        return self._to_response_dict(classification)
    
    async def get_classifications_by_tier(
        self,
        config_id: int,
        tier: Optional[str] = None
    ) -> list[dict]:
        """Obtém classificações por tier."""
        tier_enum = CampaignTier(tier) if tier else None
        
        classifications = await self.ml_repo.get_classifications_by_tier(
            config_id, tier_enum
        )
        
        return [self._to_response_dict(c) for c in classifications]
    
    async def get_classification_summary(self, config_id: int) -> dict:
        """
        Obtém resumo das classificações.
        
        Returns:
            Dict com contagens por tier e mudanças
        """
        # Obter todas as classificações válidas
        all_classifications = await self.ml_repo.get_classifications_by_tier(config_id)
        
        by_tier = {
            "HIGH_PERFORMER": 0,
            "MODERATE": 0,
            "LOW": 0,
            "UNDERPERFORMER": 0,
        }
        
        improved = 0
        declined = 0
        stable = 0
        
        for c in all_classifications:
            by_tier[c.tier.value] += 1
            
            if c.tier_change_direction == "improved":
                improved += 1
            elif c.tier_change_direction == "declined":
                declined += 1
            else:
                stable += 1
        
        return {
            "config_id": config_id,
            "total": len(all_classifications),
            "by_tier": by_tier,
            "changes": {
                "improved": improved,
                "declined": declined,
                "stable": stable,
            },
            "high_performers_count": by_tier["HIGH_PERFORMER"],
            "attention_needed": by_tier["LOW"] + by_tier["UNDERPERFORMER"],
        }
    
    async def train_classifier(
        self,
        config_id: int,
        min_samples: int = 30
    ) -> Optional[dict]:
        """
        Treina o classificador com dados históricos.
        
        Args:
            config_id: ID da configuração
            min_samples: Mínimo de campanhas para treinar
            
        Returns:
            Dict com métricas de treinamento ou None se dados insuficientes
        """
        logger.info(
            "Iniciando treinamento do classificador",
            config_id=config_id,
            min_samples=min_samples
        )
        
        # Obter features de todas as campanhas (incluindo pausadas)
        features_list = await self.data_service.get_all_campaign_features(
            config_id, active_only=False
        )
        
        if len(features_list) < min_samples:
            logger.warning(
                "Dados insuficientes para treinamento",
                available=len(features_list),
                required=min_samples
            )
            return None
        
        # Obter métricas de referência
        avg_metrics = await self.data_service.get_aggregated_metrics(config_id, days=30)
        avg_cpl = avg_metrics.get('avg_cpl', 50.0)
        avg_ctr = avg_metrics.get('avg_ctr', 1.0)
        
        # Criar labels usando heurísticas
        labels = create_training_labels(features_list, avg_cpl)
        
        # Treinar
        metrics = self.classifier.train(
            features_list, labels, avg_cpl, avg_ctr
        )
        
        # Salvar modelo
        model_path = f"{settings.models_storage_path}/classifier_config{config_id}_v{datetime.now().strftime('%Y%m%d%H%M%S')}.joblib"
        Path(settings.models_storage_path).mkdir(parents=True, exist_ok=True)
        self.classifier.save(model_path)
        
        # Registrar no banco
        model_record = await self.ml_repo.create_model(
            name=f"campaign_classifier_config{config_id}",
            model_type=ModelType.CAMPAIGN_CLASSIFIER,
            version=self.classifier.model_version,
            model_path=model_path,
            config_id=config_id,
            parameters={
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
            },
            feature_columns=self.classifier.FEATURE_COLUMNS,
        )
        
        # Atualizar com métricas
        await self.ml_repo.update_model_status(
            model_record.id,
            ModelStatus.READY,
            training_metrics=metrics,
            validation_metrics={
                "accuracy": metrics["accuracy"],
                "f1_weighted": metrics["f1_weighted"],
            }
        )
        
        # Ativar modelo
        await self.ml_repo.activate_model(
            model_record.id, ModelType.CAMPAIGN_CLASSIFIER
        )
        
        logger.info(
            "Classificador treinado e salvo",
            model_id=model_record.id,
            accuracy=metrics["accuracy"]
        )
        
        return {
            "model_id": model_record.id,
            "config_id": config_id,
            "samples_used": len(features_list),
            "metrics": metrics,
        }
    
    async def _load_active_model(self, config_id: int) -> bool:
        """
        Carrega modelo ativo do banco.
        
        Returns:
            True se modelo carregado com sucesso
        """
        model = await self.ml_repo.get_active_model(
            ModelType.CAMPAIGN_CLASSIFIER, config_id
        )
        
        if not model:
            logger.debug("Nenhum modelo ativo encontrado", config_id=config_id)
            return False
        
        try:
            self.classifier.load(model.model_path)
            
            # Atualizar last_used_at
            model.last_used_at = datetime.utcnow()
            await self.session.flush()
            
            return True
        except Exception as e:
            logger.error(
                "Erro ao carregar modelo",
                model_id=model.id,
                path=model.model_path,
                error=str(e)
            )
            return False
    
    def _to_response_dict(self, classification: MLCampaignClassification) -> dict:
        """Converte modelo para dict de resposta."""
        return {
            "id": classification.id,
            "config_id": classification.config_id,
            "campaign_id": classification.campaign_id,
            "tier": classification.tier.value,
            "confidence_score": classification.confidence_score,
            "metrics_snapshot": classification.metrics_snapshot,
            "feature_importances": classification.feature_importances,
            "previous_tier": classification.previous_tier.value if classification.previous_tier else None,
            "tier_change_direction": classification.tier_change_direction,
            "classified_at": classification.classified_at.isoformat() if classification.classified_at else None,
            "valid_until": classification.valid_until.isoformat() if classification.valid_until else None,
        }


async def get_classification_service(session: AsyncSession) -> ClassificationService:
    """Factory para criar ClassificationService."""
    return ClassificationService(session)
