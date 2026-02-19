"""
Adapter para os services FB Ads existentes.

Funcoes utilitarias para buscar e atualizar dados de campanhas
usando os services existentes em projects/facebook_ads/.

Convencao de valores monetarios:
  - DB (daily_budget): armazena em REAIS (Numeric 15,2). Ex: R$50.00 = 50.00.
    O sync de campanhas (sync_campaigns._parse_budget) converte centavos→reais
    ao importar da Graph API.
  - Graph API (daily_budget): espera CENTAVOS (int). Ex: R$50.00 = 5000.
  - Este modulo: recebe em REAIS, converte para CENTAVOS antes de enviar.
"""

import structlog
from sqlalchemy import select

from shared.db.session import async_session_maker
from shared.db.models import SistemaFacebookAdsCampaigns, SistemaFacebookAdsConfig
from projects.facebook_ads.client.base import FacebookGraphClient
from projects.facebook_ads.security.token_encryption import decrypt_token

logger = structlog.get_logger()


async def get_campaign(campaign_id: str):
    """Busca campanha pelo ID do Facebook.

    Args:
        campaign_id: ID da campanha no Facebook (ex: "23851234567890").

    Returns:
        Instancia do modelo SistemaFacebookAdsCampaigns ou None.
    """
    async with async_session_maker() as session:
        result = await session.execute(
            select(SistemaFacebookAdsCampaigns)
            .where(SistemaFacebookAdsCampaigns.campaign_id == campaign_id)
            .limit(1)
        )
        return result.scalar_one_or_none()


async def get_campaign_by_config(campaign_id: str, config_id: int):
    """Busca campanha pelo ID do Facebook E config_id (ownership).

    Args:
        campaign_id: ID da campanha no Facebook (ex: "23851234567890").
        config_id: ID da config interna (FK para sistema_facebook_ads_config.id).

    Returns:
        Instancia do modelo SistemaFacebookAdsCampaigns ou None.
    """
    async with async_session_maker() as session:
        result = await session.execute(
            select(SistemaFacebookAdsCampaigns)
            .where(
                SistemaFacebookAdsCampaigns.campaign_id == campaign_id,
                SistemaFacebookAdsCampaigns.config_id == config_id,
            )
            .limit(1)
        )
        return result.scalar_one_or_none()


async def get_campaign_budget(campaign_id: str) -> float:
    """Retorna budget atual da campanha.

    Args:
        campaign_id: ID da campanha no Facebook.

    Returns:
        Budget diario como float. 0.0 se campanha nao encontrada.
    """
    campaign = await get_campaign(campaign_id)
    return float(campaign.daily_budget or 0) if campaign else 0.0


async def _get_graph_client(campaign_id: str) -> tuple[FacebookGraphClient, str]:
    """Resolve config da campanha e retorna client autenticado.

    JOIN campaigns + config para obter access_token e account_id.

    Args:
        campaign_id: ID da campanha no Facebook.

    Returns:
        Tupla (FacebookGraphClient, account_id).

    Raises:
        ValueError: Se campanha ou config nao encontrada/inativa.
    """
    async with async_session_maker() as session:
        result = await session.execute(
            select(SistemaFacebookAdsCampaigns, SistemaFacebookAdsConfig)
            .join(
                SistemaFacebookAdsConfig,
                SistemaFacebookAdsCampaigns.config_id == SistemaFacebookAdsConfig.id,
            )
            .where(
                SistemaFacebookAdsCampaigns.campaign_id == campaign_id,
                SistemaFacebookAdsConfig.is_active.is_(True),
            )
            .limit(1)
        )
        row = result.one_or_none()
        if not row:
            raise ValueError(
                f"Campanha {campaign_id} nao encontrada ou config inativa."
            )
        campaign, config = row.tuple()

    access_token = decrypt_token(config.access_token)
    account_id = config.account_id
    client = FacebookGraphClient(access_token=access_token, account_id=account_id)
    return client, account_id


async def update_budget(campaign_id: str, new_daily_budget: float):
    """Atualiza budget via Facebook Graph API.

    Converte reais para centavos (Facebook espera centavos).

    Args:
        campaign_id: ID da campanha no Facebook.
        new_daily_budget: Novo budget diario em reais.
    """
    client, _ = await _get_graph_client(campaign_id)
    try:
        centavos = round(new_daily_budget * 100)
        await client.post(
            endpoint=campaign_id,
            data={"daily_budget": centavos},
        )
        logger.info(
            "Budget atualizado via Graph API",
            campaign_id=campaign_id,
            new_daily_budget=new_daily_budget,
            centavos=centavos,
        )
    finally:
        await client.close()


_ALLOWED_STATUSES = {"ACTIVE", "PAUSED"}


async def update_status(campaign_id: str, new_status: str):
    """Atualiza status (ACTIVE/PAUSED) via Facebook Graph API.

    Args:
        campaign_id: ID da campanha no Facebook.
        new_status: Novo status ("ACTIVE" ou "PAUSED").

    Raises:
        ValueError: Se new_status nao for ACTIVE ou PAUSED.
    """
    if new_status not in _ALLOWED_STATUSES:
        raise ValueError(
            f"Status invalido: {new_status}. Permitidos: {_ALLOWED_STATUSES}"
        )
    client, _ = await _get_graph_client(campaign_id)
    try:
        await client.post(
            endpoint=campaign_id,
            data={"status": new_status},
        )
        logger.info(
            "Status atualizado via Graph API",
            campaign_id=campaign_id,
            new_status=new_status,
        )
    finally:
        await client.close()
