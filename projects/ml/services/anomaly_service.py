"""
Serviço de detecção de anomalias.
Orquestra a detecção de anomalias em campanhas do Facebook Ads.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from shared.core.logging import get_logger
from projects.ml.db.models import AnomalySeverity
from projects.ml.db.repositories.ml_repo import MLRepository
from projects.ml.db.repositories.insights_repo import InsightsRepository
from projects.ml.algorithms.models.anomaly.anomaly_detector import (
    get_anomaly_detector,
    DetectedAnomaly,
    SeverityLevel,
)
from projects.ml.services.data_service import DataService

logger = get_logger(__name__)


@dataclass
class AnomalyDetectionResult:
    """Resultado da deteccao de anomalias."""
    config_id: int
    entity_type: str
    entities_analyzed: int
    anomalies_detected: int
    by_severity: dict[str, int]
    by_type: dict[str, int]
    anomalies: list[dict]


class AnomalyService:
    """
    Serviço para detecção e gestão de anomalias.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.ml_repo = MLRepository(session)
        self.insights_repo = InsightsRepository(session)
        self.data_service = DataService(session)
    
    def _get_entity_id(self, entity, entity_type: str) -> str:
        """Extrai entity_id de uma entidade."""
        if entity_type == "campaign":
            return entity.campaign_id
        elif entity_type == "adset":
            return entity.adset_id
        elif entity_type == "ad":
            return entity.ad_id
        return str(entity.id)

    async def _get_entities_for_analysis(
        self,
        config_id: int,
        entity_type: str,
        entity_ids: list[str]
    ) -> list:
        """Obtem entidades especificas para analise."""
        all_entities = await self.insights_repo.get_all_entities(
            config_id=config_id,
            entity_type=entity_type
        )
        # Filter by provided IDs
        id_set = set(entity_ids)
        return [e for e in all_entities if self._get_entity_id(e, entity_type) in id_set]

    async def detect_anomalies(
        self,
        config_id: int,
        entity_type: str = "campaign",
        entity_ids: Optional[list[str]] = None,
        days_to_analyze: int = 1,
        history_days: int = 30,
    ) -> AnomalyDetectionResult:
        """
        Executa deteccao de anomalias em entidades.

        Args:
            config_id: ID da configuracao FB Ads
            entity_type: 'campaign', 'adset', ou 'ad'
            entity_ids: Lista de entidades especificas (ou todas se None)
            days_to_analyze: Quantos dias recentes analisar
            history_days: Dias de historico para baseline

        Returns:
            Resultado da deteccao com anomalias encontradas
        """
        logger.info(
            "Iniciando deteccao de anomalias",
            config_id=config_id,
            entity_type=entity_type,
            days_to_analyze=days_to_analyze,
            history_days=history_days
        )

        # Buscar entidades usando o metodo generico
        if entity_ids:
            # Filter specific entities
            entities = await self._get_entities_for_analysis(
                config_id, entity_type, entity_ids
            )
        else:
            entities = await self.insights_repo.get_active_entities(
                config_id=config_id,
                entity_type=entity_type
            )

        if not entities:
            logger.info(
                "Nenhuma entidade encontrada",
                config_id=config_id,
                entity_type=entity_type
            )
            return AnomalyDetectionResult(
                config_id=config_id,
                entity_type=entity_type,
                entities_analyzed=0,
                anomalies_detected=0,
                by_severity={},
                by_type={},
                anomalies=[]
            )

        all_anomalies = []
        entities_analyzed = 0

        # Create detector once (cache is shared across entities)
        detector = get_anomaly_detector()

        # Analisar cada entidade
        for entity in entities:
            entity_id = self._get_entity_id(entity, entity_type)

            try:
                # Buscar dados historicos usando o metodo generico
                df = await self.data_service.get_entity_daily_data(
                    config_id=config_id,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    days=history_days
                )

                if df.empty:
                    continue

                entities_analyzed += 1

                # Detectar anomalias (using shared detector instance)
                anomalies = detector.detect_anomalies(
                    df=df,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    config_id=config_id,
                )

                # Salvar anomalias no banco
                for anomaly in anomalies:
                    saved = await self._save_anomaly(config_id, anomaly)
                    if saved:
                        all_anomalies.append(anomaly.to_dict())

            except Exception as e:
                logger.error(
                    "Erro ao analisar entidade",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    error=str(e)
                )
                continue

        # Commit das anomalias
        await self.session.commit()

        # Estatisticas
        by_severity = {}
        by_type = {}

        for a in all_anomalies:
            sev = a['severity']
            by_severity[sev] = by_severity.get(sev, 0) + 1

            atype = a['anomaly_type']
            by_type[atype] = by_type.get(atype, 0) + 1

        logger.info(
            "Deteccao de anomalias concluida",
            config_id=config_id,
            entity_type=entity_type,
            entities_analyzed=entities_analyzed,
            anomalies_detected=len(all_anomalies)
        )

        return AnomalyDetectionResult(
            config_id=config_id,
            entity_type=entity_type,
            entities_analyzed=entities_analyzed,
            anomalies_detected=len(all_anomalies),
            by_severity=by_severity,
            by_type=by_type,
            anomalies=all_anomalies
        )
    
    async def _save_anomaly(
        self,
        config_id: int,
        anomaly: DetectedAnomaly,
    ) -> bool:
        """
        Salva anomalia no banco de dados.
        Evita duplicatas verificando se já existe anomalia similar recente não resolvida.
        """
        try:
            # Mapear severidade
            severity_map = {
                SeverityLevel.LOW: AnomalySeverity.LOW,
                SeverityLevel.MEDIUM: AnomalySeverity.MEDIUM,
                SeverityLevel.HIGH: AnomalySeverity.HIGH,
                SeverityLevel.CRITICAL: AnomalySeverity.CRITICAL,
            }

            db_severity = severity_map.get(anomaly.severity, AnomalySeverity.MEDIUM)

            # Verificar duplicata (mesma campanha, tipo, métrica NÃO ACKNOWLEDGED nas últimas 6 horas)
            # Mudança: Buscar apenas anomalias não reconhecidas (is_acknowledged=False)
            # Mudança: Reduzir window de 24h para 6h (task roda a cada 1h)
            from datetime import datetime, timedelta
            cutoff_time = datetime.utcnow() - timedelta(hours=6)

            existing = await self.ml_repo.get_anomalies(
                config_id=config_id,
                acknowledged=False,  # ← Apenas não reconhecidas
                start_date=cutoff_time  # ← Últimas 6 horas apenas
            )
            
            for e in existing:
                if (e.entity_id == anomaly.entity_id and
                    e.anomaly_type == anomaly.anomaly_type.value and
                    e.metric_name == anomaly.metric_name):
                    logger.debug(
                        "Anomalia duplicada ignorada (já existe não resolvida nas últimas 6h)",
                        entity_id=anomaly.entity_id,
                        anomaly_type=anomaly.anomaly_type.value,
                        existing_id=e.id,
                        detected_at=e.detected_at.isoformat()
                    )
                    return False
            
            # Obter nomes da entidade para identificação visual
            entity_names = await self.data_service.get_entity_names(
                config_id, anomaly.entity_type, anomaly.entity_id
            )

            # Criar anomalia
            created_anomaly = await self.ml_repo.create_anomaly(
                config_id=config_id,
                entity_type=anomaly.entity_type,
                entity_id=anomaly.entity_id,
                anomaly_type=anomaly.anomaly_type.value,
                metric_name=anomaly.metric_name,
                observed_value=anomaly.observed_value,
                expected_value=anomaly.expected_value,
                deviation_score=anomaly.deviation_score,
                severity=db_severity,
                anomaly_date=anomaly.anomaly_date,
                campaign_name=entity_names.get('campaign_name'),
                adset_name=entity_names.get('adset_name'),
                ad_name=entity_names.get('ad_name'),
            )

            logger.info(
                "Anomalia salva com sucesso",
                entity_id=anomaly.entity_id,
                anomaly_type=anomaly.anomaly_type.value,
                severity=db_severity.value,
                metric=anomaly.metric_name,
                deviation=round(anomaly.deviation_score, 2)
            )

            return True
            
        except Exception as e:
            logger.error(
                "Erro ao salvar anomalia",
                entity_id=anomaly.entity_id,
                error=str(e)
            )
            return False
    
    async def get_anomaly_stats(
        self,
        config_id: int,
        days: int = 7,
    ) -> dict:
        """
        Obtém estatísticas de anomalias do período.
        """
        anomalies = await self.ml_repo.get_anomalies(
            config_id=config_id,
            days=days
        )
        
        stats = {
            'total': len(anomalies),
            'unacknowledged': sum(1 for a in anomalies if not a.is_acknowledged),
            'by_severity': {},
            'by_type': {},
            'by_entity': {},
            'critical_count': 0,
        }
        
        for a in anomalies:
            # Por severidade
            sev = a.severity.value
            stats['by_severity'][sev] = stats['by_severity'].get(sev, 0) + 1
            
            if a.severity == AnomalySeverity.CRITICAL:
                stats['critical_count'] += 1
            
            # Por tipo
            stats['by_type'][a.anomaly_type] = stats['by_type'].get(a.anomaly_type, 0) + 1
            
            # Por entidade
            if a.entity_id not in stats['by_entity']:
                stats['by_entity'][a.entity_id] = 0
            stats['by_entity'][a.entity_id] += 1
        
        return stats
