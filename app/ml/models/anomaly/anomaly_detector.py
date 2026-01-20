"""
Detector de anomalias para métricas de Facebook Ads.

Utiliza múltiplos métodos:
1. Z-Score para detecção de outliers estatísticos
2. IQR (Interquartile Range) para distribuições não-normais
3. Isolation Forest para detecção multivariada (quando treinado)
4. Mudanças abruptas (change-point detection)
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from app.core.logging import get_logger

logger = get_logger(__name__)


class AnomalyType(str, Enum):
    """Tipos de anomalias detectáveis."""
    SPEND_SPIKE = "spend_spike"
    SPEND_DROP = "spend_drop"
    CPL_SPIKE = "cpl_spike"
    CPL_DROP = "cpl_drop"
    CTR_DROP = "ctr_drop"
    CTR_SPIKE = "ctr_spike"
    PERFORMANCE_DROP = "performance_drop"
    PERFORMANCE_SPIKE = "performance_spike"
    FREQUENCY_ALERT = "frequency_alert"
    REACH_SATURATION = "reach_saturation"
    ZERO_SPEND = "zero_spend"
    ZERO_IMPRESSIONS = "zero_impressions"


class SeverityLevel(str, Enum):
    """Níveis de severidade."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class DetectedAnomaly:
    """Anomalia detectada."""
    entity_type: str
    entity_id: str
    anomaly_type: AnomalyType
    metric_name: str
    observed_value: float
    expected_value: float
    deviation_score: float  # Z-score ou similar
    severity: SeverityLevel
    anomaly_date: datetime
    description: str
    
    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "anomaly_type": self.anomaly_type.value,
            "metric_name": self.metric_name,
            "observed_value": self.observed_value,
            "expected_value": self.expected_value,
            "deviation_score": self.deviation_score,
            "severity": self.severity.value,
            "anomaly_date": self.anomaly_date,
            "description": self.description,
        }


class AnomalyDetector:
    """
    Detector de anomalias baseado em métodos estatísticos.
    
    Configurações:
    - z_threshold: Threshold do Z-score para considerar anomalia (padrão: 2.5)
    - iqr_multiplier: Multiplicador do IQR (padrão: 1.5)
    - min_history_days: Mínimo de dias de histórico necessários (padrão: 7)
    """
    
    def __init__(
        self,
        z_threshold: float = 2.5,
        iqr_multiplier: float = 1.5,
        min_history_days: int = 7,
    ):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.min_history_days = min_history_days
    
    def detect_anomalies(
        self,
        df: pd.DataFrame,
        entity_type: str,
        entity_id: str,
        analysis_date: Optional[datetime] = None,
    ) -> list[DetectedAnomaly]:
        """
        Detecta anomalias em um DataFrame de métricas diárias.
        
        Args:
            df: DataFrame com colunas: date, spend, impressions, clicks, leads, etc.
            entity_type: Tipo da entidade (campaign, adset, ad)
            entity_id: ID da entidade
            analysis_date: Data para análise (padrão: última data no df)
            
        Returns:
            Lista de anomalias detectadas
        """
        if df.empty or len(df) < self.min_history_days:
            logger.debug(
                "Dados insuficientes para detecção de anomalias",
                entity_id=entity_id,
                rows=len(df)
            )
            return []
        
        anomalies = []
        
        # Ordenar por data
        df = df.sort_values('date').reset_index(drop=True)
        
        # Data de análise
        if analysis_date is None:
            analysis_date = df['date'].max()
        
        # Filtrar dados até a data de análise
        df_analysis = df[df['date'] <= analysis_date].copy()
        
        if len(df_analysis) < self.min_history_days:
            return []
        
        # Calcular métricas derivadas
        df_analysis = self._calculate_derived_metrics(df_analysis)
        
        # Valor atual (última linha) vs histórico
        current = df_analysis.iloc[-1]
        history = df_analysis.iloc[:-1]
        
        # 1. Detectar anomalias em SPEND
        spend_anomaly = self._detect_metric_anomaly(
            current['spend'],
            history['spend'],
            'spend',
            entity_type,
            entity_id,
            analysis_date
        )
        if spend_anomaly:
            anomalies.append(spend_anomaly)
        
        # 2. Detectar anomalias em CPL (se houver leads)
        if 'cpl' in df_analysis.columns:
            cpl_history = history[history['cpl'].notna() & (history['cpl'] > 0)]['cpl']
            if len(cpl_history) >= 5 and current.get('cpl', 0) > 0:
                cpl_anomaly = self._detect_metric_anomaly(
                    current['cpl'],
                    cpl_history,
                    'cpl',
                    entity_type,
                    entity_id,
                    analysis_date
                )
                if cpl_anomaly:
                    anomalies.append(cpl_anomaly)
        
        # 3. Detectar anomalias em CTR
        if 'ctr' in df_analysis.columns:
            ctr_history = history[history['ctr'].notna()]['ctr']
            if len(ctr_history) >= 5:
                ctr_anomaly = self._detect_metric_anomaly(
                    current.get('ctr', 0),
                    ctr_history,
                    'ctr',
                    entity_type,
                    entity_id,
                    analysis_date
                )
                if ctr_anomaly:
                    anomalies.append(ctr_anomaly)
        
        # 4. Detectar zero spend (campanha parada)
        if current['spend'] == 0 and history['spend'].mean() > 10:
            anomalies.append(DetectedAnomaly(
                entity_type=entity_type,
                entity_id=entity_id,
                anomaly_type=AnomalyType.ZERO_SPEND,
                metric_name='spend',
                observed_value=0,
                expected_value=float(history['spend'].mean()),
                deviation_score=-999,
                severity=SeverityLevel.HIGH,
                anomaly_date=analysis_date,
                description=f"Campanha sem gasto. Média histórica: R$ {history['spend'].mean():.2f}"
            ))
        
        # 5. Detectar frequency alta (fadiga de audiência)
        if 'frequency' in df_analysis.columns:
            freq = current.get('frequency', 0)
            if freq > 3.0:
                severity = SeverityLevel.LOW
                if freq > 5.0:
                    severity = SeverityLevel.MEDIUM
                if freq > 7.0:
                    severity = SeverityLevel.HIGH
                if freq > 10.0:
                    severity = SeverityLevel.CRITICAL
                
                anomalies.append(DetectedAnomaly(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    anomaly_type=AnomalyType.FREQUENCY_ALERT,
                    metric_name='frequency',
                    observed_value=freq,
                    expected_value=2.5,  # Valor ideal
                    deviation_score=freq / 2.5,
                    severity=severity,
                    anomaly_date=analysis_date,
                    description=f"Frequência alta ({freq:.1f}). Risco de fadiga de audiência."
                ))
        
        # 6. Detectar mudanças abruptas (últimos 3 dias vs anteriores)
        change_anomalies = self._detect_change_points(
            df_analysis,
            entity_type,
            entity_id,
            analysis_date
        )
        anomalies.extend(change_anomalies)
        
        logger.info(
            "Detecção de anomalias concluída",
            entity_id=entity_id,
            anomalies_found=len(anomalies)
        )
        
        return anomalies
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula métricas derivadas como CPL, CTR, CPC."""
        df = df.copy()
        
        # CTR (Click-Through Rate)
        df['ctr'] = np.where(
            df['impressions'] > 0,
            df['clicks'] / df['impressions'] * 100,
            0
        )
        
        # CPL (Cost Per Lead)
        df['cpl'] = np.where(
            df['leads'] > 0,
            df['spend'] / df['leads'],
            np.nan
        )
        
        # CPC (Cost Per Click)
        df['cpc'] = np.where(
            df['clicks'] > 0,
            df['spend'] / df['clicks'],
            np.nan
        )
        
        return df
    
    def _detect_metric_anomaly(
        self,
        current_value: float,
        history: pd.Series,
        metric_name: str,
        entity_type: str,
        entity_id: str,
        analysis_date: datetime,
    ) -> Optional[DetectedAnomaly]:
        """
        Detecta anomalia em uma métrica específica usando Z-score e IQR.
        """
        if len(history) < 5:
            return None
        
        # Remover outliers extremos do histórico para cálculo
        clean_history = history[~((history - history.mean()).abs() > 4 * history.std())]
        
        if len(clean_history) < 5:
            clean_history = history
        
        mean = clean_history.mean()
        std = clean_history.std()
        
        if std == 0:
            return None
        
        # Z-Score
        z_score = (current_value - mean) / std
        
        # IQR check
        q1, q3 = clean_history.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        is_anomaly = abs(z_score) > self.z_threshold or current_value < lower_bound or current_value > upper_bound
        
        if not is_anomaly:
            return None
        
        # Determinar tipo e severidade
        anomaly_type, severity = self._classify_anomaly(
            metric_name,
            z_score,
            current_value,
            mean
        )
        
        # Descrição
        direction = "acima" if z_score > 0 else "abaixo"
        description = (
            f"{metric_name.upper()} {direction} do esperado. "
            f"Valor: {current_value:.2f}, Média: {mean:.2f}, "
            f"Desvio: {abs(z_score):.1f}σ"
        )
        
        return DetectedAnomaly(
            entity_type=entity_type,
            entity_id=entity_id,
            anomaly_type=anomaly_type,
            metric_name=metric_name,
            observed_value=current_value,
            expected_value=mean,
            deviation_score=z_score,
            severity=severity,
            anomaly_date=analysis_date,
            description=description
        )
    
    def _classify_anomaly(
        self,
        metric_name: str,
        z_score: float,
        current_value: float,
        mean: float
    ) -> tuple[AnomalyType, SeverityLevel]:
        """Classifica o tipo e severidade da anomalia."""
        
        abs_z = abs(z_score)
        
        # Determinar severidade baseada no Z-score
        if abs_z < 3:
            severity = SeverityLevel.LOW
        elif abs_z < 4:
            severity = SeverityLevel.MEDIUM
        elif abs_z < 5:
            severity = SeverityLevel.HIGH
        else:
            severity = SeverityLevel.CRITICAL
        
        # Determinar tipo baseado na métrica e direção
        is_spike = z_score > 0
        
        type_mapping = {
            'spend': (AnomalyType.SPEND_SPIKE, AnomalyType.SPEND_DROP),
            'cpl': (AnomalyType.CPL_SPIKE, AnomalyType.CPL_DROP),
            'ctr': (AnomalyType.CTR_SPIKE, AnomalyType.CTR_DROP),
        }
        
        if metric_name in type_mapping:
            spike_type, drop_type = type_mapping[metric_name]
            anomaly_type = spike_type if is_spike else drop_type
        else:
            anomaly_type = AnomalyType.PERFORMANCE_SPIKE if is_spike else AnomalyType.PERFORMANCE_DROP
        
        # Ajustar severidade para métricas específicas
        # CPL alto é mais grave
        if metric_name == 'cpl' and is_spike:
            if severity == SeverityLevel.LOW:
                severity = SeverityLevel.MEDIUM
            elif severity == SeverityLevel.MEDIUM:
                severity = SeverityLevel.HIGH
        
        # CTR baixo é preocupante
        if metric_name == 'ctr' and not is_spike:
            if severity == SeverityLevel.LOW:
                severity = SeverityLevel.MEDIUM
        
        return anomaly_type, severity
    
    def _detect_change_points(
        self,
        df: pd.DataFrame,
        entity_type: str,
        entity_id: str,
        analysis_date: datetime,
    ) -> list[DetectedAnomaly]:
        """
        Detecta mudanças abruptas comparando últimos 3 dias vs 7 anteriores.
        """
        anomalies = []
        
        if len(df) < 10:
            return anomalies
        
        recent = df.tail(3)
        previous = df.iloc[-10:-3]
        
        if len(previous) < 5:
            return anomalies
        
        for metric in ['spend', 'cpl', 'ctr']:
            if metric not in df.columns:
                continue
            
            recent_values = recent[metric].dropna()
            previous_values = previous[metric].dropna()
            
            if len(recent_values) < 2 or len(previous_values) < 3:
                continue
            
            recent_mean = recent_values.mean()
            previous_mean = previous_values.mean()
            previous_std = previous_values.std()
            
            if previous_std == 0 or previous_mean == 0:
                continue
            
            # Calcular mudança percentual
            pct_change = (recent_mean - previous_mean) / previous_mean * 100
            
            # Threshold para mudança significativa: 30%+
            if abs(pct_change) < 30:
                continue
            
            # Criar anomalia de mudança
            is_spike = pct_change > 0
            
            if metric == 'cpl':
                anomaly_type = AnomalyType.CPL_SPIKE if is_spike else AnomalyType.CPL_DROP
            elif metric == 'spend':
                anomaly_type = AnomalyType.SPEND_SPIKE if is_spike else AnomalyType.SPEND_DROP
            else:
                anomaly_type = AnomalyType.PERFORMANCE_SPIKE if is_spike else AnomalyType.PERFORMANCE_DROP
            
            severity = SeverityLevel.MEDIUM
            if abs(pct_change) > 50:
                severity = SeverityLevel.HIGH
            if abs(pct_change) > 80:
                severity = SeverityLevel.CRITICAL
            
            direction = "aumentou" if is_spike else "diminuiu"
            anomalies.append(DetectedAnomaly(
                entity_type=entity_type,
                entity_id=entity_id,
                anomaly_type=anomaly_type,
                metric_name=metric,
                observed_value=recent_mean,
                expected_value=previous_mean,
                deviation_score=pct_change / 100,
                severity=severity,
                anomaly_date=analysis_date,
                description=(
                    f"{metric.upper()} {direction} {abs(pct_change):.0f}% nos últimos 3 dias. "
                    f"De {previous_mean:.2f} para {recent_mean:.2f}."
                )
            ))
        
        return anomalies


# Instância global do detector
anomaly_detector = AnomalyDetector()
