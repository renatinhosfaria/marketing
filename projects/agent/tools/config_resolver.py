"""
Adapter layer: resolve account_id (string Facebook) para config_id (int PK).

A ML API existente usa config_id (int, PK de sistema_facebook_ads_config).
O Agent usa account_id (string do Facebook, ex: "act_123456").
Todas as tools que chamam a ML API precisam resolver account_id -> config_id.
"""

from sqlalchemy import select
from cachetools import TTLCache

from shared.db.session import async_session_maker
from shared.db.models import SistemaFacebookAdsConfig


# Cache em memoria: config_id raramente muda. TTL 5 min, max 100 contas.
_config_id_cache: TTLCache = TTLCache(maxsize=100, ttl=300)


async def resolve_config_id(account_id: str) -> int | None:
    """Resolve account_id (string Facebook) para config_id (int PK).

    Usado por todas as tools que chamam a ML API.
    Cache local com TTL de 5 minutos para evitar query repetida.

    Args:
        account_id: ID da conta do Facebook (ex: "act_123456").

    Returns:
        config_id (int) ou None se nao encontrado/inativo.
    """
    if account_id in _config_id_cache:
        return _config_id_cache[account_id]

    async with async_session_maker() as session:
        result = await session.execute(
            select(SistemaFacebookAdsConfig.id)
            .where(SistemaFacebookAdsConfig.account_id == account_id)
            .where(SistemaFacebookAdsConfig.is_active.is_(True))
            .limit(1)
        )
        config_id = result.scalar_one_or_none()

    if config_id is not None:
        _config_id_cache[account_id] = config_id
    return config_id
