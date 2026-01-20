"""
Classificador de campanhas usando XGBoost.
Categoriza campanhas em tiers de performance.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
import xgboost as xgb

from app.services.feature_engineering import CampaignFeatures, FeatureEngineer
from app.db.models.ml_models import CampaignTier
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationResult:
    """Resultado da classificação de uma campanha."""
    campaign_id: str
    config_id: int
    tier: CampaignTier
    confidence_score: float
    probabilities: dict[str, float]
    feature_importances: dict[str, float]
    metrics_snapshot: dict


class CampaignClassifier:
    """
    Classificador de campanhas usando XGBoost.
    
    Tiers:
    - HIGH_PERFORMER: CPL baixo, leads consistentes, tendências positivas
    - MODERATE: Performance aceitável, potencial de melhoria
    - LOW: Performance abaixo da média, precisa atenção
    - UNDERPERFORMER: Performance ruim, considerar pausar
    """
    
    # Features usadas para classificação
    FEATURE_COLUMNS = [
        'cpl_ratio',           # CPL / média (normalizado)
        'ctr_ratio',           # CTR / média
        'leads_7d_normalized', # Leads normalizados pelo spend
        'cpl_trend',           # Tendência do CPL
        'leads_trend',         # Tendência de leads
        'cpl_volatility',      # Volatilidade do CPL (std)
        'conversion_rate_7d',  # Taxa de conversão
        'days_with_leads_ratio', # % de dias com leads
        'frequency_score',     # Score de frequência (penaliza alta)
        'consistency_score',   # Score de consistência
    ]
    
    TIER_ORDER = [
        CampaignTier.UNDERPERFORMER,
        CampaignTier.LOW,
        CampaignTier.MODERATE,
        CampaignTier.HIGH_PERFORMER,
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa o classificador.
        
        Args:
            model_path: Caminho para modelo serializado (opcional)
        """
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_engineer = FeatureEngineer()
        self.is_fitted = False
        self.model_version = "1.0.0"
        
        # Referências para normalização
        self.avg_cpl: float = 50.0
        self.avg_ctr: float = 1.0
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def prepare_features(
        self,
        campaign_features: CampaignFeatures,
        avg_cpl: float,
        avg_ctr: float
    ) -> pd.DataFrame:
        """
        Prepara features para classificação.
        
        Args:
            campaign_features: Features da campanha
            avg_cpl: CPL médio de referência
            avg_ctr: CTR médio de referência
            
        Returns:
            DataFrame com features prontas para o modelo
        """
        self.avg_cpl = avg_cpl if avg_cpl > 0 else 50.0
        self.avg_ctr = avg_ctr if avg_ctr > 0 else 1.0
        
        # Calcular features derivadas
        cpl_ratio = campaign_features.cpl_7d / self.avg_cpl if self.avg_cpl > 0 else 1.0
        ctr_ratio = campaign_features.ctr_7d / self.avg_ctr if self.avg_ctr > 0 else 1.0
        
        # Normalizar leads pelo spend (leads por R$100 gastos)
        leads_per_100 = (campaign_features.leads_7d / campaign_features.spend_7d * 100) if campaign_features.spend_7d > 0 else 0
        
        # Score de frequência (penalizar alta frequência)
        frequency_score = max(0, 1 - (campaign_features.frequency_7d - 1) / 3) if campaign_features.frequency_7d > 0 else 1.0
        
        # Score de consistência
        days_with_leads_ratio = campaign_features.days_with_leads_7d / 7
        
        # Volatilidade normalizada
        cpl_volatility = campaign_features.cpl_std_7d / self.avg_cpl if self.avg_cpl > 0 else 0
        
        # Consistency score composto
        consistency_score = (
            days_with_leads_ratio * 0.5 +
            (1 - min(abs(campaign_features.cpl_trend) / 50, 1)) * 0.3 +
            frequency_score * 0.2
        )
        
        features = {
            'cpl_ratio': min(cpl_ratio, 5.0),  # Cap em 5x
            'ctr_ratio': min(ctr_ratio, 5.0),
            'leads_7d_normalized': min(leads_per_100, 10.0),
            'cpl_trend': max(min(campaign_features.cpl_trend, 100), -100),  # Limitar
            'leads_trend': max(min(campaign_features.leads_trend, 100), -100),
            'cpl_volatility': min(cpl_volatility, 2.0),
            'conversion_rate_7d': min(campaign_features.conversion_rate_7d, 50),
            'days_with_leads_ratio': days_with_leads_ratio,
            'frequency_score': frequency_score,
            'consistency_score': consistency_score,
        }
        
        return pd.DataFrame([features])
    
    def classify_by_rules(
        self,
        campaign_features: CampaignFeatures,
        avg_cpl: float,
        avg_ctr: float
    ) -> ClassificationResult:
        """
        Classifica uma campanha usando regras (fallback quando não há modelo).
        
        Args:
            campaign_features: Features da campanha
            avg_cpl: CPL médio
            avg_ctr: CTR médio
            
        Returns:
            ClassificationResult
        """
        cpl_ratio = campaign_features.cpl_7d / avg_cpl if avg_cpl > 0 else 1.0
        
        # Classificação baseada em regras
        if campaign_features.leads_7d == 0 and campaign_features.spend_7d > 50:
            tier = CampaignTier.UNDERPERFORMER
            confidence = 0.9
        elif cpl_ratio <= 0.7 and campaign_features.days_with_leads_7d >= 4:
            tier = CampaignTier.HIGH_PERFORMER
            confidence = 0.85
        elif cpl_ratio <= 1.0 and campaign_features.leads_7d >= 2:
            tier = CampaignTier.MODERATE
            confidence = 0.75
        elif cpl_ratio <= 1.5:
            tier = CampaignTier.LOW
            confidence = 0.7
        else:
            tier = CampaignTier.UNDERPERFORMER
            confidence = 0.7
        
        # Ajustar por tendência
        if campaign_features.cpl_trend > 20:
            # CPL subindo, penalizar
            tier_idx = self.TIER_ORDER.index(tier)
            if tier_idx > 0:
                tier = self.TIER_ORDER[tier_idx - 1]
                confidence *= 0.9
        elif campaign_features.cpl_trend < -15:
            # CPL caindo, bonificar
            tier_idx = self.TIER_ORDER.index(tier)
            if tier_idx < len(self.TIER_ORDER) - 1:
                tier = self.TIER_ORDER[tier_idx + 1]
                confidence *= 0.9
        
        # Probabilidades estimadas
        tier_idx = self.TIER_ORDER.index(tier)
        probs = {t.value: 0.1 for t in self.TIER_ORDER}
        probs[tier.value] = confidence
        remaining = 1 - confidence
        for i, t in enumerate(self.TIER_ORDER):
            if t != tier:
                probs[t.value] = remaining / 3
        
        return ClassificationResult(
            campaign_id=campaign_features.campaign_id,
            config_id=campaign_features.config_id,
            tier=tier,
            confidence_score=confidence,
            probabilities=probs,
            feature_importances={},  # Sem modelo, sem importances
            metrics_snapshot={
                'cpl_7d': campaign_features.cpl_7d,
                'ctr_7d': campaign_features.ctr_7d,
                'leads_7d': campaign_features.leads_7d,
                'spend_7d': campaign_features.spend_7d,
                'cpl_trend': campaign_features.cpl_trend,
                'avg_cpl_reference': avg_cpl,
            }
        )
    
    def classify(
        self,
        campaign_features: CampaignFeatures,
        avg_cpl: float,
        avg_ctr: float
    ) -> ClassificationResult:
        """
        Classifica uma campanha.
        
        Args:
            campaign_features: Features da campanha
            avg_cpl: CPL médio de referência
            avg_ctr: CTR médio de referência
            
        Returns:
            ClassificationResult com tier e probabilidades
        """
        # Se não tiver modelo treinado, usar regras
        if not self.is_fitted or self.model is None:
            logger.debug("Usando classificação por regras (modelo não disponível)")
            return self.classify_by_rules(campaign_features, avg_cpl, avg_ctr)
        
        # Preparar features
        X = self.prepare_features(campaign_features, avg_cpl, avg_ctr)
        X_scaled = self.scaler.transform(X[self.FEATURE_COLUMNS])
        
        # Predição
        proba = self.model.predict_proba(X_scaled)[0]
        pred_idx = np.argmax(proba)
        tier_label = self.label_encoder.inverse_transform([pred_idx])[0]
        tier = CampaignTier(tier_label)
        confidence = float(proba[pred_idx])
        
        # Probabilidades por tier
        probabilities = {
            self.label_encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(proba)
        }
        
        # Feature importances
        importances = dict(zip(
            self.FEATURE_COLUMNS,
            self.model.feature_importances_.tolist()
        ))
        
        return ClassificationResult(
            campaign_id=campaign_features.campaign_id,
            config_id=campaign_features.config_id,
            tier=tier,
            confidence_score=confidence,
            probabilities=probabilities,
            feature_importances=importances,
            metrics_snapshot={
                'cpl_7d': campaign_features.cpl_7d,
                'ctr_7d': campaign_features.ctr_7d,
                'leads_7d': campaign_features.leads_7d,
                'spend_7d': campaign_features.spend_7d,
                'cpl_trend': campaign_features.cpl_trend,
                'avg_cpl_reference': avg_cpl,
            }
        )
    
    def train(
        self,
        features_list: list[CampaignFeatures],
        labels: list[CampaignTier],
        avg_cpl: float,
        avg_ctr: float,
        test_size: float = 0.2,
    ) -> dict:
        """
        Treina o classificador com dados históricos.
        
        Args:
            features_list: Lista de CampaignFeatures
            labels: Lista de tiers correspondentes
            avg_cpl: CPL médio de referência
            avg_ctr: CTR médio de referência
            test_size: Proporção para teste
            
        Returns:
            Dict com métricas de treinamento
        """
        logger.info(
            "Iniciando treinamento do classificador",
            samples=len(features_list)
        )
        
        self.avg_cpl = avg_cpl
        self.avg_ctr = avg_ctr
        
        # Preparar dados
        X_list = []
        for features in features_list:
            X_df = self.prepare_features(features, avg_cpl, avg_ctr)
            X_list.append(X_df)
        
        X = pd.concat(X_list, ignore_index=True)
        y = [t.value for t in labels]
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X[self.FEATURE_COLUMNS], y_encoded,
            test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=4,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
        )
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        self.is_fitted = True
        
        metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'samples_train': len(X_train),
            'samples_test': len(X_test),
            'feature_importances': dict(zip(
                self.FEATURE_COLUMNS,
                self.model.feature_importances_.tolist()
            )),
        }
        
        logger.info(
            "Treinamento concluído",
            accuracy=accuracy,
            f1_weighted=f1_weighted
        )
        
        return metrics
    
    def save(self, path: str) -> None:
        """Salva o modelo e componentes."""
        if not self.is_fitted:
            raise ValueError("Modelo não treinado")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.FEATURE_COLUMNS,
            'avg_cpl': self.avg_cpl,
            'avg_ctr': self.avg_ctr,
            'version': self.model_version,
            'saved_at': datetime.utcnow().isoformat(),
        }
        
        joblib.dump(model_data, path)
        logger.info("Modelo salvo", path=path)
    
    def load(self, path: str) -> None:
        """Carrega modelo de arquivo."""
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.avg_cpl = model_data.get('avg_cpl', 50.0)
        self.avg_ctr = model_data.get('avg_ctr', 1.0)
        self.model_version = model_data.get('version', '1.0.0')
        self.is_fitted = True
        
        logger.info("Modelo carregado", path=path, version=self.model_version)


# Funções auxiliares para criar labels de treinamento
def create_training_labels(
    features_list: list[CampaignFeatures],
    avg_cpl: float
) -> list[CampaignTier]:
    """
    Cria labels de treinamento baseado em regras heurísticas.
    Usado para treinamento inicial quando não há labels manuais.
    """
    labels = []
    
    for features in features_list:
        cpl_ratio = features.cpl_7d / avg_cpl if avg_cpl > 0 else 1.0
        
        # Heurísticas para labeling
        if features.leads_7d == 0 and features.spend_7d > 50:
            label = CampaignTier.UNDERPERFORMER
        elif cpl_ratio <= 0.7 and features.days_with_leads_7d >= 4:
            label = CampaignTier.HIGH_PERFORMER
        elif cpl_ratio <= 1.0 and features.leads_7d >= 2:
            label = CampaignTier.MODERATE
        elif cpl_ratio <= 1.5:
            label = CampaignTier.LOW
        else:
            label = CampaignTier.UNDERPERFORMER
        
        labels.append(label)
    
    return labels


# Instância global
classifier = CampaignClassifier()
