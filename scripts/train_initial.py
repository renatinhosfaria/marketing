#!/usr/bin/env python
"""
Script para treinamento inicial dos modelos ML.
Deve ser executado após init_db.py e com dados suficientes no banco.
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import sessionmaker
from shared.db.session import sync_engine
from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
from projects.ml.db.models import MLTrainedModel, ModelType, ModelStatus
from shared.core.logging import setup_logging, get_logger

setup_logging("INFO")
logger = get_logger(__name__)

Session = sessionmaker(bind=sync_engine)


def get_active_configs():
    """Obtém configurações ativas de Facebook Ads."""
    session = Session()
    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()
        return [(c.id, c.name) for c in configs]
    finally:
        session.close()


def count_insights(config_id: int) -> int:
    """Conta registros de insights para uma configuração."""
    from shared.db.models.famachat_readonly import SistemaFacebookAdsInsightsHistory

    session = Session()
    try:
        count = session.query(SistemaFacebookAdsInsightsHistory).filter(
            SistemaFacebookAdsInsightsHistory.config_id == config_id
        ).count()
        return count
    finally:
        session.close()


def main():
    """Função principal."""
    logger.info("=" * 60)
    logger.info("Marketing - Treinamento Inicial")
    logger.info("=" * 60)

    # Obter configs
    configs = get_active_configs()
    if not configs:
        logger.warning("Nenhuma configuração de Facebook Ads ativa encontrada")
        logger.info("Adicione configurações de Facebook Ads antes de treinar modelos")
        return

    logger.info(f"Encontradas {len(configs)} configurações ativas")

    for config_id, name in configs:
        logger.info(f"\n--- Config: {name} (ID: {config_id}) ---")

        # Contar dados disponíveis
        insights_count = count_insights(config_id)
        logger.info(f"Registros de insights: {insights_count}")

        if insights_count < 30:
            logger.warning(
                f"Dados insuficientes para treinamento. "
                f"Mínimo: 30, disponível: {insights_count}"
            )
            continue

        logger.info("Dados suficientes para treinamento!")
        logger.info("TODO: Implementar treinamento real nas fases 3-6")

        # Placeholder para treinamento
        # train_recommender(config_id)      # Fase 3
        # train_classifier(config_id)       # Fase 4
        # train_forecasters(config_id)      # Fase 5
        # train_anomaly_detector(config_id) # Fase 6

    logger.info("\n" + "=" * 60)
    logger.info("Treinamento inicial concluído")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
