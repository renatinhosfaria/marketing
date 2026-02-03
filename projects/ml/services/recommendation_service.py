"""
Serviço de recomendações de otimização.
Orquestra a geração, armazenamento e gestão de recomendações.
Suporta múltiplos níveis: campaign, adset, ad.
"""

from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from projects.ml.db.repositories.ml_repo import MLRepository
from projects.ml.db.models import (
    MLRecommendation,
    RecommendationType,
)
from projects.ml.services.data_service import DataService
from projects.ml.services.feature_engineering import CampaignFeatures, EntityFeatures
from projects.ml.algorithms.models.recommendation.rule_engine import (
    RuleEngine,
    Recommendation,
    create_rule_engine,
)
from shared.core.logging import get_logger

logger = get_logger(__name__)

# Valid entity types
VALID_ENTITY_TYPES = {"campaign", "adset", "ad"}


class RecommendationService:
    """
    Serviço para gerenciar recomendações de otimização.
    Suporta múltiplos níveis: campaign, adset, ad.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.ml_repo = MLRepository(session)
        self.data_service = DataService(session)

    async def generate_entity_recommendations(
        self,
        config_id: int,
        entity_type: str = "campaign",
        entity_ids: Optional[list[str]] = None,
        force_refresh: bool = False
    ) -> list[dict]:
        """
        Gera recomendações para entidades de uma configuração.

        Args:
            config_id: ID da configuração FB Ads
            entity_type: Tipo de entidade ("campaign", "adset", "ad")
            entity_ids: IDs específicos ou None para todas
            force_refresh: Se True, gera mesmo se houver recomendações ativas

        Returns:
            Lista de recomendações geradas
        """
        if entity_type not in VALID_ENTITY_TYPES:
            raise ValueError(f"entity_type deve ser um de: {VALID_ENTITY_TYPES}")

        logger.info(
            "Iniciando geração de recomendações",
            config_id=config_id,
            entity_type=entity_type,
            entity_ids=entity_ids,
            force_refresh=force_refresh
        )

        # Verificação de dados desabilitada temporariamente
        logger.info("Verificação de dados desabilitada temporariamente", config_id=config_id)

        # Obter métricas agregadas de referência
        avg_metrics = await self.data_service.get_aggregated_metrics(config_id, days=14)

        # Criar rule engine
        rule_engine = create_rule_engine(avg_metrics)

        # Obter features das entidades
        if entity_ids:
            features_list = []
            for eid in entity_ids:
                features = await self.data_service.get_entity_features(
                    config_id, entity_type, eid
                )
                if features:
                    features_list.append(features)
        else:
            features_list = await self.data_service.get_all_entity_features(
                config_id, entity_type, active_only=False  # Incluir pausadas para REACTIVATE
            )

        if not features_list:
            logger.info(
                f"Nenhum(a) {entity_type} com features disponíveis",
                config_id=config_id,
                entity_type=entity_type
            )
            return []

        # Se não forçar refresh, verificar recomendações existentes
        existing_entity_ids = set()
        if not force_refresh:
            existing = await self.ml_repo.get_active_recommendations(
                config_id, entity_type=entity_type
            )
            existing_entity_ids = {r.entity_id for r in existing}

        # Gerar recomendações
        all_recommendations = []

        for features in features_list:
            entity_id = self._get_entity_id(features, entity_type)

            # Pular se já tem recomendações ativas (exceto se force_refresh)
            if entity_id in existing_entity_ids:
                continue

            # Obter classificação atual se disponível
            classification = await self.ml_repo.get_latest_classification(
                config_id, entity_id, entity_type
            )
            tier = classification.tier.value if classification else None

            # Calcular dias no tier atual
            days_in_tier = 0
            if classification:
                days_in_tier = (datetime.utcnow() - classification.classified_at).days

            # Obter features de irmãos para comparação (apenas para ads)
            sibling_features = None
            if entity_type == "ad":
                parent_id = getattr(features, 'parent_id', None)
                if parent_id:
                    sibling_features = await self._get_sibling_features(
                        config_id, entity_type, parent_id, entity_id
                    )

            # Gerar recomendações via rule engine
            recommendations = rule_engine.generate_recommendations(
                features,
                tier,
                days_in_tier,
                entity_type=entity_type,
                sibling_features=sibling_features
            )

            for rec in recommendations:
                all_recommendations.append(rec)

        # Salvar no banco
        saved_recommendations = []
        for rec in all_recommendations:
            saved = await self._save_recommendation(rec)
            if saved:
                saved_recommendations.append(self._to_response_dict(saved))

        logger.info(
            "Recomendações geradas",
            config_id=config_id,
            entity_type=entity_type,
            total_entities=len(features_list),
            recommendations_generated=len(saved_recommendations)
        )

        return saved_recommendations

    async def generate_recommendations(
        self,
        config_id: int,
        campaign_ids: Optional[list[str]] = None,
        force_refresh: bool = False
    ) -> list[dict]:
        """
        Gera recomendações para campanhas de uma configuração.
        Método de compatibilidade - delega para generate_entity_recommendations.

        Args:
            config_id: ID da configuração FB Ads
            campaign_ids: IDs específicos ou None para todas
            force_refresh: Se True, gera mesmo se houver recomendações ativas

        Returns:
            Lista de recomendações geradas
        """
        return await self.generate_entity_recommendations(
            config_id=config_id,
            entity_type="campaign",
            entity_ids=campaign_ids,
            force_refresh=force_refresh
        )
    
    async def get_recommendations(
        self,
        config_id: int,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        recommendation_type: Optional[str] = None,
        active_only: bool = True,
        limit: int = 50
    ) -> list[dict]:
        """
        Obtém recomendações filtradas.
        
        Args:
            config_id: ID da configuração
            entity_type: Filtrar por tipo de entidade
            entity_id: Filtrar por ID da entidade
            recommendation_type: Filtrar por tipo de recomendação
            active_only: Apenas recomendações ativas
            limit: Limite de resultados
            
        Returns:
            Lista de recomendações
        """
        recommendations = await self.ml_repo.get_recommendations(
            config_id=config_id,
            entity_type=entity_type,
            entity_id=entity_id,
            recommendation_type=RecommendationType(recommendation_type) if recommendation_type else None,
            active_only=active_only,
            limit=limit
        )
        
        return [self._to_response_dict(r) for r in recommendations]
    
    async def get_campaign_recommendations(
        self,
        config_id: int,
        campaign_id: str
    ) -> list[dict]:
        """Obtém recomendações de uma campanha específica."""
        return await self.get_recommendations(
            config_id=config_id,
            entity_type="campaign",
            entity_id=campaign_id,
            active_only=True
        )
    
    async def get_recommendation_summary(
        self,
        config_id: int,
        entity_type: Optional[str] = None
    ) -> dict:
        """
        Obtém resumo das recomendações ativas.

        Args:
            config_id: ID da configuração
            entity_type: Filtrar por tipo de entidade (opcional)

        Returns:
            Dict com contagens por tipo e prioridade
        """
        if entity_type and entity_type not in VALID_ENTITY_TYPES:
            raise ValueError(f"entity_type deve ser um de: {VALID_ENTITY_TYPES}")

        recommendations = await self.ml_repo.get_active_recommendations(
            config_id, entity_type=entity_type
        )

        by_type = {}
        by_priority = {"high": 0, "medium": 0, "low": 0}
        by_entity_type = {"campaign": 0, "adset": 0, "ad": 0}

        for rec in recommendations:
            # Por tipo de recomendação
            type_name = rec.recommendation_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

            # Por prioridade
            if rec.priority >= 7:
                by_priority["high"] += 1
            elif rec.priority >= 4:
                by_priority["medium"] += 1
            else:
                by_priority["low"] += 1

            # Por tipo de entidade
            if rec.entity_type in by_entity_type:
                by_entity_type[rec.entity_type] += 1

        return {
            "config_id": config_id,
            "entity_type_filter": entity_type,
            "total": len(recommendations),
            "by_type": by_type,
            "by_priority": by_priority,
            "by_entity_type": by_entity_type,
            "high_priority_count": by_priority["high"],
        }
    
    async def dismiss_recommendation(
        self,
        recommendation_id: int,
        user_id: int,
        reason: Optional[str] = None
    ) -> Optional[dict]:
        """
        Marca uma recomendação como descartada.
        
        Args:
            recommendation_id: ID da recomendação
            user_id: ID do usuário que descartou
            reason: Motivo do descarte
            
        Returns:
            Recomendação atualizada ou None
        """
        recommendation = await self.ml_repo.get_recommendation(recommendation_id)
        if not recommendation:
            return None
        
        await self.ml_repo.dismiss_recommendation(
            recommendation_id, user_id, reason
        )
        
        # Recarregar
        recommendation = await self.ml_repo.get_recommendation(recommendation_id)
        
        logger.info(
            "Recomendação descartada",
            recommendation_id=recommendation_id,
            user_id=user_id,
            reason=reason
        )
        
        return self._to_response_dict(recommendation) if recommendation else None
    
    async def apply_recommendation(
        self,
        recommendation_id: int,
        user_id: int
    ) -> Optional[dict]:
        """
        Marca uma recomendação como aplicada.
        
        Args:
            recommendation_id: ID da recomendação
            user_id: ID do usuário que aplicou
            
        Returns:
            Recomendação atualizada ou None
        """
        recommendation = await self.ml_repo.get_recommendation(recommendation_id)
        if not recommendation:
            return None
        
        await self.ml_repo.apply_recommendation(recommendation_id, user_id)
        
        # Recarregar
        recommendation = await self.ml_repo.get_recommendation(recommendation_id)
        
        logger.info(
            "Recomendação aplicada",
            recommendation_id=recommendation_id,
            user_id=user_id,
            type=recommendation.recommendation_type.value if recommendation else None
        )
        
        return self._to_response_dict(recommendation) if recommendation else None
    
    async def expire_old_recommendations(self, config_id: int) -> int:
        """
        Expira recomendações antigas.
        
        Returns:
            Número de recomendações expiradas
        """
        count = await self.ml_repo.expire_recommendations(config_id)
        logger.info(
            "Recomendações expiradas",
            config_id=config_id,
            count=count
        )
        return count
    
    def _get_entity_id(self, features, entity_type: str) -> str:
        """Extrai o ID da entidade a partir das features."""
        if isinstance(features, EntityFeatures):
            return features.entity_id
        elif isinstance(features, CampaignFeatures):
            return features.campaign_id
        # Fallback para atributos diretos
        if entity_type == "campaign":
            return getattr(features, 'campaign_id', getattr(features, 'entity_id', ''))
        elif entity_type == "adset":
            return getattr(features, 'adset_id', getattr(features, 'entity_id', ''))
        elif entity_type == "ad":
            return getattr(features, 'ad_id', getattr(features, 'entity_id', ''))
        return getattr(features, 'entity_id', '')

    async def _get_sibling_features(
        self,
        config_id: int,
        entity_type: str,
        parent_id: str,
        exclude_entity_id: str
    ) -> list:
        """
        Obtém features de entidades irmãs (mesmo parent_id).
        Usado para comparação em regras como CREATIVE_WINNER.
        """
        all_features = await self.data_service.get_all_entity_features(
            config_id, entity_type, active_only=True
        )

        siblings = []
        for f in all_features:
            f_parent_id = getattr(f, 'parent_id', None)
            f_entity_id = self._get_entity_id(f, entity_type)
            if f_parent_id == parent_id and f_entity_id != exclude_entity_id:
                siblings.append(f)

        return siblings

    async def _save_recommendation(self, rec: Recommendation) -> Optional[MLRecommendation]:
        """Salva uma recomendação no banco."""
        try:
            start_of_day = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            exists = await self.ml_repo.recommendation_exists(
                config_id=rec.config_id,
                entity_type=rec.entity_type,
                entity_id=rec.entity_id,
                recommendation_type=rec.recommendation_type,
                since=start_of_day,
            )
            if exists:
                logger.debug(
                    "Recomendacao duplicada ignorada",
                    config_id=rec.config_id,
                    entity_id=rec.entity_id,
                    recommendation_type=rec.recommendation_type.value,
                )
                return None

            expires_at = datetime.utcnow() + timedelta(days=rec.expires_in_days)

            saved = await self.ml_repo.create_recommendation(
                config_id=rec.config_id,
                entity_type=rec.entity_type,
                entity_id=rec.entity_id,
                recommendation_type=rec.recommendation_type,
                priority=rec.priority,
                title=rec.title,
                description=rec.description,
                suggested_action=rec.suggested_action,
                confidence_score=rec.confidence_score,
                reasoning=rec.reasoning,
                expires_at=expires_at
            )
            
            return saved
        except Exception as e:
            logger.error(
                "Erro ao salvar recomendação",
                error=str(e),
                campaign_id=rec.entity_id
            )
            return None
    
    def _to_response_dict(self, rec: MLRecommendation) -> dict:
        """Converte modelo para dict de resposta."""
        return {
            "id": rec.id,
            "config_id": rec.config_id,
            "entity_type": rec.entity_type,
            "entity_id": rec.entity_id,
            "recommendation_type": rec.recommendation_type.value,
            "priority": rec.priority,
            "title": rec.title,
            "description": rec.description,
            "suggested_action": rec.suggested_action,
            "confidence_score": rec.confidence_score,
            "reasoning": rec.reasoning,
            "is_active": rec.is_active,
            "was_applied": rec.was_applied,
            "applied_at": rec.applied_at.isoformat() if rec.applied_at else None,
            "dismissed": rec.dismissed,
            "dismissed_at": rec.dismissed_at.isoformat() if rec.dismissed_at else None,
            "dismissed_reason": rec.dismissed_reason,
            "created_at": rec.created_at.isoformat() if rec.created_at else None,
            "expires_at": rec.expires_at.isoformat() if rec.expires_at else None,
        }


async def get_recommendation_service(session: AsyncSession) -> RecommendationService:
    """Factory para criar RecommendationService."""
    return RecommendationService(session)
