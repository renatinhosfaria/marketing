"""
Criar tabelas ML iniciais.

Revision ID: 20241211_001
Revises: 
Create Date: 2024-12-11

Tabelas criadas:
- ml_trained_models: Modelos ML treinados
- ml_predictions: Previsões de CPL/Leads
- ml_campaign_classifications: Classificação de campanhas
- ml_recommendations: Recomendações de otimização
- ml_anomalies: Anomalias detectadas
- ml_training_jobs: Jobs de treinamento
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20241211_001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Criar todas as tabelas ML usando SQL puro para evitar conflitos de ENUMs."""
    
    # ==================== ENUMS ====================
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE modeltype AS ENUM ('TIME_SERIES_CPL', 'TIME_SERIES_LEADS', 'CAMPAIGN_CLASSIFIER', 'ANOMALY_DETECTOR', 'RECOMMENDER');
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$;
    """)
    
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE modelstatus AS ENUM ('TRAINING', 'READY', 'ACTIVE', 'DEPRECATED', 'FAILED');
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$;
    """)
    
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE predictiontype AS ENUM ('CPL_FORECAST', 'LEADS_FORECAST', 'SPEND_FORECAST');
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$;
    """)
    
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE campaigntier AS ENUM ('HIGH_PERFORMER', 'MODERATE', 'LOW', 'UNDERPERFORMER');
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$;
    """)
    
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE recommendationtype AS ENUM ('BUDGET_INCREASE', 'BUDGET_DECREASE', 'PAUSE_CAMPAIGN', 'SCALE_UP', 'CREATIVE_REFRESH', 'AUDIENCE_REVIEW', 'REACTIVATE', 'OPTIMIZE_SCHEDULE');
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$;
    """)
    
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE anomalyseverity AS ENUM ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL');
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$;
    """)
    
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE jobstatus AS ENUM ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED');
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$;
    """)

    # ==================== TABELAS ====================
    
    # ml_trained_models
    op.execute("""
        CREATE TABLE IF NOT EXISTS ml_trained_models (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            model_type modeltype NOT NULL,
            version VARCHAR(50) NOT NULL,
            config_id INTEGER,
            model_path VARCHAR(500) NOT NULL,
            parameters JSONB,
            feature_columns JSONB,
            training_metrics JSONB,
            validation_metrics JSONB,
            status modelstatus DEFAULT 'TRAINING',
            is_active BOOLEAN DEFAULT FALSE,
            training_data_start TIMESTAMP,
            training_data_end TIMESTAMP,
            samples_count INTEGER,
            created_at TIMESTAMP DEFAULT NOW(),
            trained_at TIMESTAMP,
            last_used_at TIMESTAMP
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_ml_trained_models_type_active ON ml_trained_models (model_type, is_active);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_ml_trained_models_config_type ON ml_trained_models (config_id, model_type);")
    
    # ml_predictions
    op.execute("""
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id SERIAL PRIMARY KEY,
            model_id INTEGER REFERENCES ml_trained_models(id) ON DELETE SET NULL,
            config_id INTEGER NOT NULL,
            entity_type VARCHAR(50) NOT NULL,
            entity_id VARCHAR(100) NOT NULL,
            prediction_type predictiontype NOT NULL,
            forecast_date TIMESTAMP NOT NULL,
            horizon_days INTEGER NOT NULL,
            predicted_value FLOAT NOT NULL,
            confidence_lower FLOAT,
            confidence_upper FLOAT,
            actual_value FLOAT,
            absolute_error FLOAT,
            percentage_error FLOAT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_ml_predictions_entity ON ml_predictions (entity_type, entity_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_ml_predictions_config_date ON ml_predictions (config_id, forecast_date);")
    
    # ml_campaign_classifications
    op.execute("""
        CREATE TABLE IF NOT EXISTS ml_campaign_classifications (
            id SERIAL PRIMARY KEY,
            config_id INTEGER NOT NULL,
            campaign_id VARCHAR(100) NOT NULL,
            tier campaigntier NOT NULL,
            confidence_score FLOAT NOT NULL,
            metrics_snapshot JSONB,
            feature_importances JSONB,
            previous_tier campaigntier,
            tier_change_direction VARCHAR(20),
            classified_at TIMESTAMP DEFAULT NOW(),
            valid_until TIMESTAMP
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_ml_classifications_config_campaign ON ml_campaign_classifications (config_id, campaign_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_ml_classifications_tier ON ml_campaign_classifications (tier);")
    
    # ml_recommendations
    op.execute("""
        CREATE TABLE IF NOT EXISTS ml_recommendations (
            id SERIAL PRIMARY KEY,
            config_id INTEGER NOT NULL,
            entity_type VARCHAR(50) NOT NULL,
            entity_id VARCHAR(100) NOT NULL,
            recommendation_type recommendationtype NOT NULL,
            priority INTEGER NOT NULL,
            title VARCHAR(255) NOT NULL,
            description TEXT,
            suggested_action JSONB,
            confidence_score FLOAT NOT NULL,
            reasoning JSONB,
            is_active BOOLEAN DEFAULT TRUE,
            was_applied BOOLEAN DEFAULT FALSE,
            applied_at TIMESTAMP,
            applied_by INTEGER,
            dismissed BOOLEAN DEFAULT FALSE,
            dismissed_at TIMESTAMP,
            dismissed_by INTEGER,
            dismissed_reason TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            expires_at TIMESTAMP
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_ml_recommendations_config_active ON ml_recommendations (config_id, is_active);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_ml_recommendations_entity ON ml_recommendations (entity_type, entity_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_ml_recommendations_type ON ml_recommendations (recommendation_type);")
    
    # ml_anomalies
    op.execute("""
        CREATE TABLE IF NOT EXISTS ml_anomalies (
            id SERIAL PRIMARY KEY,
            config_id INTEGER NOT NULL,
            entity_type VARCHAR(50) NOT NULL,
            entity_id VARCHAR(100) NOT NULL,
            anomaly_type VARCHAR(100) NOT NULL,
            metric_name VARCHAR(100) NOT NULL,
            observed_value FLOAT NOT NULL,
            expected_value FLOAT NOT NULL,
            deviation_score FLOAT NOT NULL,
            severity anomalyseverity NOT NULL,
            is_acknowledged BOOLEAN DEFAULT FALSE,
            acknowledged_by INTEGER,
            acknowledged_at TIMESTAMP,
            resolution_notes TEXT,
            anomaly_date TIMESTAMP NOT NULL,
            detected_at TIMESTAMP DEFAULT NOW()
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_ml_anomalies_config_date ON ml_anomalies (config_id, anomaly_date);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_ml_anomalies_severity ON ml_anomalies (severity);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_ml_anomalies_entity ON ml_anomalies (entity_type, entity_id);")
    
    # ml_training_jobs
    op.execute("""
        CREATE TABLE IF NOT EXISTS ml_training_jobs (
            id SERIAL PRIMARY KEY,
            model_type modeltype NOT NULL,
            config_id INTEGER,
            celery_task_id VARCHAR(255),
            status jobstatus DEFAULT 'PENDING',
            progress FLOAT DEFAULT 0.0,
            model_id INTEGER REFERENCES ml_trained_models(id) ON DELETE SET NULL,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_ml_training_jobs_status ON ml_training_jobs (status);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_ml_training_jobs_config_type ON ml_training_jobs (config_id, model_type);")


def downgrade() -> None:
    """Remover todas as tabelas ML."""
    op.execute("DROP TABLE IF EXISTS ml_training_jobs CASCADE;")
    op.execute("DROP TABLE IF EXISTS ml_anomalies CASCADE;")
    op.execute("DROP TABLE IF EXISTS ml_recommendations CASCADE;")
    op.execute("DROP TABLE IF EXISTS ml_campaign_classifications CASCADE;")
    op.execute("DROP TABLE IF EXISTS ml_predictions CASCADE;")
    op.execute("DROP TABLE IF EXISTS ml_trained_models CASCADE;")
    
    op.execute("DROP TYPE IF EXISTS jobstatus;")
    op.execute("DROP TYPE IF EXISTS anomalyseverity;")
    op.execute("DROP TYPE IF EXISTS recommendationtype;")
    op.execute("DROP TYPE IF EXISTS campaigntier;")
    op.execute("DROP TYPE IF EXISTS predictiontype;")
    op.execute("DROP TYPE IF EXISTS modelstatus;")
    op.execute("DROP TYPE IF EXISTS modeltype;")
