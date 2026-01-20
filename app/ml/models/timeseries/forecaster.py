"""
Forecaster de séries temporais para métricas de Facebook Ads.

Implementa previsões de CPL e Leads usando múltiplos métodos:
1. Média móvel exponencial (EMA)
2. Tendência linear
3. Prophet (se disponível)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Literal
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from app.core.logging import get_logger

logger = get_logger(__name__)

# Tentar importar Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.info("Prophet não disponível - usando métodos alternativos")


@dataclass
class ForecastResult:
    """Resultado de uma previsão."""
    entity_type: str
    entity_id: str
    metric: str
    forecast_date: datetime
    predicted_value: float
    confidence_lower: float
    confidence_upper: float
    confidence_level: float  # 0.95 para 95%
    method: str  # 'prophet', 'ema', 'linear'
    model_version: str


@dataclass
class ForecastSeries:
    """Série de previsões."""
    entity_type: str
    entity_id: str
    metric: str
    forecasts: list[ForecastResult]
    historical: list[dict]  # Últimos N dias históricos
    method: str
    created_at: datetime


class TimeSeriesForecaster:
    """
    Forecaster de séries temporais para métricas de ads.

    Métricas suportadas:
    - CPL (Custo por Lead)
    - Leads
    - Spend

    Métodos:
    - Prophet (se disponível)
    - EMA (Média Móvel Exponencial)
    - Linear (Regressão Linear)
    """

    def __init__(
        self,
        method: Literal['auto', 'prophet', 'ema', 'linear'] = 'auto',
        confidence_level: float = 0.95,
    ):
        """
        Inicializa o forecaster.

        Args:
            method: Método de previsão ('auto' usa Prophet se disponível)
            confidence_level: Nível de confiança para intervalos (0.95 = 95%)
        """
        self.confidence_level = confidence_level
        self.model_version = "1.0.0"

        # Selecionar método
        if method == 'auto':
            self.method = 'prophet' if PROPHET_AVAILABLE else 'ema'
        else:
            if method == 'prophet' and not PROPHET_AVAILABLE:
                logger.warning("Prophet não disponível, usando EMA")
                self.method = 'ema'
            else:
                self.method = method

        logger.info(f"Forecaster inicializado com método: {self.method}")

    def forecast(
        self,
        df: pd.DataFrame,
        metric: str,
        entity_type: str,
        entity_id: str,
        horizon_days: int = 7,
    ) -> ForecastSeries:
        """
        Gera previsões para uma métrica.

        Args:
            df: DataFrame com colunas 'date' e a métrica (ex: 'cpl', 'leads', 'spend')
            metric: Nome da métrica a prever
            entity_type: Tipo da entidade (campaign, adset, ad)
            entity_id: ID da entidade
            horizon_days: Dias de previsão (1-30)

        Returns:
            ForecastSeries com previsões e histórico
        """
        if df.empty or len(df) < 7:
            raise ValueError(f"Dados insuficientes para previsão. Mínimo: 7 dias, encontrado: {len(df)}")

        if metric not in df.columns:
            raise ValueError(f"Métrica '{metric}' não encontrada no DataFrame")

        # Preparar dados
        df = df.sort_values('date').reset_index(drop=True)
        df = df[['date', metric]].dropna()

        if len(df) < 7:
            raise ValueError(f"Dados insuficientes após remover NaN. Mínimo: 7, encontrado: {len(df)}")

        # Escolher método
        if self.method == 'prophet' and PROPHET_AVAILABLE:
            forecasts = self._forecast_prophet(df, metric, entity_type, entity_id, horizon_days)
        elif self.method == 'linear':
            forecasts = self._forecast_linear(df, metric, entity_type, entity_id, horizon_days)
        else:
            forecasts = self._forecast_ema(df, metric, entity_type, entity_id, horizon_days)

        # Preparar histórico
        historical = df.tail(14).to_dict('records')
        for h in historical:
            h['date'] = h['date'].isoformat() if isinstance(h['date'], datetime) else str(h['date'])

        return ForecastSeries(
            entity_type=entity_type,
            entity_id=entity_id,
            metric=metric,
            forecasts=forecasts,
            historical=historical,
            method=self.method,
            created_at=datetime.utcnow()
        )

    def _forecast_prophet(
        self,
        df: pd.DataFrame,
        metric: str,
        entity_type: str,
        entity_id: str,
        horizon_days: int,
    ) -> list[ForecastResult]:
        """Previsão usando Prophet."""
        # Preparar dados no formato Prophet
        prophet_df = df.rename(columns={'date': 'ds', metric: 'y'})

        # Configurar modelo
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                interval_width=self.confidence_level,
            )
            model.fit(prophet_df)

        # Gerar previsões
        future = model.make_future_dataframe(periods=horizon_days)
        forecast = model.predict(future)

        # Extrair previsões futuras
        last_date = df['date'].max()
        future_forecast = forecast[forecast['ds'] > last_date]

        results = []
        for _, row in future_forecast.iterrows():
            results.append(ForecastResult(
                entity_type=entity_type,
                entity_id=entity_id,
                metric=metric,
                forecast_date=row['ds'].to_pydatetime(),
                predicted_value=max(0, float(row['yhat'])),  # Não negativos
                confidence_lower=max(0, float(row['yhat_lower'])),
                confidence_upper=max(0, float(row['yhat_upper'])),
                confidence_level=self.confidence_level,
                method='prophet',
                model_version=self.model_version
            ))

        return results

    def _forecast_ema(
        self,
        df: pd.DataFrame,
        metric: str,
        entity_type: str,
        entity_id: str,
        horizon_days: int,
    ) -> list[ForecastResult]:
        """
        Previsão usando Média Móvel Exponencial.

        Combina tendência com EMA para previsões mais estáveis.
        """
        values = df[metric].values
        dates = pd.to_datetime(df['date'])
        last_date = dates.max()

        # Calcular EMA com spans diferentes
        ema_short = pd.Series(values).ewm(span=3, adjust=False).mean().iloc[-1]
        ema_medium = pd.Series(values).ewm(span=7, adjust=False).mean().iloc[-1]
        ema_long = pd.Series(values).ewm(span=14, adjust=False).mean().iloc[-1]

        # Tendência baseada em EMA
        trend = (ema_short - ema_long) / 14

        # Volatilidade para intervalos de confiança
        std = np.std(values[-14:]) if len(values) >= 14 else np.std(values)
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)

        results = []
        base_value = (ema_short * 0.5 + ema_medium * 0.3 + ema_long * 0.2)  # Média ponderada

        for i in range(1, horizon_days + 1):
            forecast_date = last_date + timedelta(days=i)

            # Valor previsto com tendência atenuada
            predicted = base_value + (trend * i * 0.5)
            predicted = max(0, predicted)

            # Intervalos de confiança aumentam com o horizonte
            uncertainty = std * z_score * np.sqrt(i)
            lower = max(0, predicted - uncertainty)
            upper = predicted + uncertainty

            results.append(ForecastResult(
                entity_type=entity_type,
                entity_id=entity_id,
                metric=metric,
                forecast_date=forecast_date,
                predicted_value=float(predicted),
                confidence_lower=float(lower),
                confidence_upper=float(upper),
                confidence_level=self.confidence_level,
                method='ema',
                model_version=self.model_version
            ))

        return results

    def _forecast_linear(
        self,
        df: pd.DataFrame,
        metric: str,
        entity_type: str,
        entity_id: str,
        horizon_days: int,
    ) -> list[ForecastResult]:
        """Previsão usando Regressão Linear."""
        values = df[metric].values
        dates = pd.to_datetime(df['date'])
        last_date = dates.max()

        # Usar apenas últimos 14 dias para tendência
        n_points = min(14, len(values))
        x = np.arange(n_points)
        y = values[-n_points:]

        # Regressão linear
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Volatilidade residual
        predictions = slope * x + intercept
        residuals = y - predictions
        residual_std = np.std(residuals)
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)

        results = []
        for i in range(1, horizon_days + 1):
            forecast_date = last_date + timedelta(days=i)

            # Previsão linear
            predicted = slope * (n_points + i - 1) + intercept
            predicted = max(0, predicted)

            # Intervalos de confiança
            uncertainty = residual_std * z_score * np.sqrt(1 + 1/n_points + ((i)**2) / np.var(x))
            lower = max(0, predicted - uncertainty)
            upper = predicted + uncertainty

            results.append(ForecastResult(
                entity_type=entity_type,
                entity_id=entity_id,
                metric=metric,
                forecast_date=forecast_date,
                predicted_value=float(predicted),
                confidence_lower=float(lower),
                confidence_upper=float(upper),
                confidence_level=self.confidence_level,
                method='linear',
                model_version=self.model_version
            ))

        return results

    def forecast_cpl(
        self,
        df: pd.DataFrame,
        entity_type: str,
        entity_id: str,
        horizon_days: int = 7,
    ) -> ForecastSeries:
        """
        Gera previsões de CPL.

        Args:
            df: DataFrame com colunas 'date', 'spend', 'leads'
            entity_type: Tipo da entidade
            entity_id: ID da entidade
            horizon_days: Dias de previsão

        Returns:
            ForecastSeries com previsões de CPL
        """
        # Calcular CPL se não existir
        if 'cpl' not in df.columns:
            df = df.copy()
            df['cpl'] = np.where(
                df['leads'] > 0,
                df['spend'] / df['leads'],
                np.nan
            )

        return self.forecast(df, 'cpl', entity_type, entity_id, horizon_days)

    def forecast_leads(
        self,
        df: pd.DataFrame,
        entity_type: str,
        entity_id: str,
        horizon_days: int = 7,
    ) -> ForecastSeries:
        """
        Gera previsões de leads.

        Args:
            df: DataFrame com colunas 'date', 'leads'
            entity_type: Tipo da entidade
            entity_id: ID da entidade
            horizon_days: Dias de previsão

        Returns:
            ForecastSeries com previsões de leads
        """
        return self.forecast(df, 'leads', entity_type, entity_id, horizon_days)

    def forecast_spend(
        self,
        df: pd.DataFrame,
        entity_type: str,
        entity_id: str,
        horizon_days: int = 7,
    ) -> ForecastSeries:
        """
        Gera previsões de spend.

        Args:
            df: DataFrame com colunas 'date', 'spend'
            entity_type: Tipo da entidade
            entity_id: ID da entidade
            horizon_days: Dias de previsão

        Returns:
            ForecastSeries com previsões de spend
        """
        return self.forecast(df, 'spend', entity_type, entity_id, horizon_days)


# Instância global do forecaster
forecaster = TimeSeriesForecaster(method='auto')
