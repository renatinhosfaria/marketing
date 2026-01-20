"""
Serviço de detecção de anomalias.
Orquestra a detecção de anomalias em campanhas do Facebook Ads.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models.ml_models import AnomalySeverity
from app.db.repositories.ml_repo import MLRepository
from app.db.repositories.insights_repo import InsightsRepository
from app.ml.models.anomaly.anomaly_detector import (
    anomaly_detector,
    DetectedAnomaly,
    SeverityLevel,
)
from app.services.data_service import DataService

logger = get_logger(__name__)


@dataclass
class AnomalyDetectionResult:
    """Resultado da detecção de anomalias."""
    config_id: int
    campaigns_analyzed: int
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
    
    async def detect_anomalies(
        self,
        config_id: int,
        campaign_ids: Optional[list[str]] = None,
        days_to_analyze: int = 1,
        history_days: int = 30,
    ) -> AnomalyDetectionResult:
        """
        Executa detecção de anomalias em campanhas.
        
        Args:
            config_id: ID da configuração FB Ads
            campaign_ids: Lista de campanhas específicas (ou todas se None)
            days_to_analyze: Quantos dias recentes analisar
            history_days: Dias de histórico para baseline
            
        Returns:
            Resultado da detecção com anomalias encontradas
        """
        logger.info(
            "Iniciando detecção de anomalias",
            config_id=config_id,
            days_to_analyze=days_to_analyze,
            history_days=history_days
        )
        
        # Buscar campanhas
        if campaign_ids:
            campaigns = await self.insights_repo.get_campaigns_by_ids(
                config_id=config_id,
                campaign_ids=campaign_ids
            )
        else:
            campaigns = await self.insights_repo.get_active_campaigns(
                config_id=config_id
            )
        
        if not campaigns:
            logger.info("Nenhuma campanha encontrada", config_id=config_id)
            return AnomalyDetectionResult(
                config_id=config_id,
                campaigns_analyzed=0,
                anomalies_detected=0,
                by_severity={},
                by_type={},
                anomalies=[]
            )
        
        all_anomalies = []
        campaigns_analyzed = 0
        
        # Analisar cada campanha
        for campaign in campaigns:
            campaign_id = campaign.campaign_id
            
            try:
                # Buscar dados históricos
                df = await self.data_service.get_campaign_daily_data(
                    config_id=config_id,
                    campaign_id=campaign_id,
                    days=history_days
                )
                
                if df.empty:
                    continue
                
                campaigns_analyzed += 1
                
                # Detectar anomalias
                anomalies = anomaly_detector.detect_anomalies(
                    df=df,
                    entity_type='campaign',
                    entity_id=campaign_id,
                )
                
                # Salvar anomalias no banco
                for anomaly in anomalies:
                    saved = await self._save_anomaly(config_id, anomaly)
                    if saved:
                        all_anomalies.append(anomaly.to_dict())
                
            except Exception as e:
                logger.error(
                    "Erro ao analisar campanha",
                    campaign_id=campaign_id,
                    error=str(e)
                )
                continue
        
        # Commit das anomalias
        await self.session.commit()
        
        # Estatísticas
        by_severity = {}
        by_type = {}
        
        for a in all_anomalies:
            sev = a['severity']
            by_severity[sev] = by_severity.get(sev, 0) + 1
            
            atype = a['anomaly_type']
            by_type[atype] = by_type.get(atype, 0) + 1
        
        logger.info(
            "Detecção de anomalias concluída",
            config_id=config_id,
            campaigns_analyzed=campaigns_analyzed,
            anomalies_detected=len(all_anomalies)
        )
        
        return AnomalyDetectionResult(
            config_id=config_id,
            campaigns_analyzed=campaigns_analyzed,
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
        Evita duplicatas verificando se já existe anomalia similar recente.
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
            
            # Verificar duplicata (mesma campanha, tipo, métrica no mesmo dia)
            existing = await self.ml_repo.get_anomalies(
                config_id=config_id,
                days=1
            )
            
            for e in existing:
                if (e.entity_id == anomaly.entity_id and 
                    e.anomaly_type == anomaly.anomaly_type.value and
                    e.metric_name == anomaly.metric_name):
                    logger.debug(
                        "Anomalia duplicada ignorada",
                        entity_id=anomaly.entity_id,
                        anomaly_type=anomaly.anomaly_type.value
                    )
                    return False
            
            # Criar anomalia
            await self.ml_repo.create_anomaly(
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
