#!/usr/bin/env python
"""
Script para inicializar as tabelas de ML no banco de dados.
Cria todas as tabelas definidas nos modelos SQLAlchemy.
"""

import sys
import os

# Adicionar path do app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from shared.db.session import sync_engine, Base
from projects.ml.db.models import (
    MLTrainedModel,
    MLPrediction,
    MLCampaignClassification,
    MLRecommendation,
    MLAnomaly,
    MLTrainingJob,
    MLFeature,
    MLForecast,
)
from shared.core.logging import setup_logging, get_logger

setup_logging("INFO")
logger = get_logger(__name__)


def check_connection():
    """Verifica conexão com o banco de dados."""
    try:
        with sync_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Conexão com banco de dados OK")
        return True
    except Exception as e:
        logger.error(f"Erro ao conectar ao banco: {e}")
        return False


def create_ml_tables():
    """Cria as tabelas de ML no banco de dados."""
    logger.info("Criando tabelas de ML...")

    try:
        # Criar apenas as tabelas de ML (não as read-only do FamaChat)
        ml_tables = [
            MLTrainedModel.__table__,
            MLPrediction.__table__,
            MLCampaignClassification.__table__,
            MLRecommendation.__table__,
            MLAnomaly.__table__,
            MLTrainingJob.__table__,
            MLFeature.__table__,
            MLForecast.__table__,
        ]

        for table in ml_tables:
            table.create(sync_engine, checkfirst=True)
            logger.info(f"Tabela criada/verificada: {table.name}")

        logger.info("Todas as tabelas de ML criadas com sucesso!")
        return True

    except Exception as e:
        logger.error(f"Erro ao criar tabelas: {e}")
        return False


def verify_readonly_tables():
    """Verifica se as tabelas read-only existem."""
    logger.info("Verificando tabelas read-only...")

    required_tables = [
        "sistema_facebook_ads_config",
        "sistema_facebook_ads_campaigns",
        "sistema_facebook_ads_adsets",
        "sistema_facebook_ads_ads",
        "sistema_facebook_ads_insights_history",
        "sistema_facebook_ads_insights_today",
    ]

    try:
        with sync_engine.connect() as conn:
            for table_name in required_tables:
                result = conn.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = '{table_name}'
                    )
                """))
                exists = result.scalar()

                if exists:
                    logger.info(f"✓ Tabela {table_name} encontrada")
                else:
                    logger.warning(f"✗ Tabela {table_name} NÃO encontrada")

        return True

    except Exception as e:
        logger.error(f"Erro ao verificar tabelas: {e}")
        return False


def main():
    """Função principal."""
    logger.info("=" * 60)
    logger.info("Marketing - Inicialização do Banco de Dados")
    logger.info("=" * 60)

    # 1. Verificar conexão
    if not check_connection():
        logger.error("Falha na conexão. Verifique DATABASE_URL no .env")
        sys.exit(1)

    # 2. Verificar tabelas read-only
    verify_readonly_tables()

    # 3. Criar tabelas de ML
    if not create_ml_tables():
        logger.error("Falha ao criar tabelas de ML")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Inicialização concluída com sucesso!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
