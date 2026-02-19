"""
Validacao de ownership para tools read-only.

Garante que entidades (campaign, adset, ad) pertencem ao config_id
do usuario autenticado. Previne leitura cross-tenant via prompt injection.
"""

from sqlalchemy import select

from shared.db.session import async_session_maker
from shared.db.models import (
    SistemaFacebookAdsCampaigns,
    SistemaFacebookAdsAdsets,
    SistemaFacebookAdsAds,
)


# Map explicito: modelo SQLAlchemy + coluna de ID por tipo de entidade.
# Usa os modelos existentes em shared.db.models (tabelas sistema_facebook_ads_*).
MODEL_BY_TYPE = {
    "campaign": (SistemaFacebookAdsCampaigns, "campaign_id"),
    "adset": (SistemaFacebookAdsAdsets, "adset_id"),
    "ad": (SistemaFacebookAdsAds, "ad_id"),
}


async def _validate_entity_ownership(
    entity_id: str,
    config_id: int,
    entity_type: str = "campaign",
) -> bool:
    """Valida que a entidade pertence a conta do usuario.

    Chamado em todas as tools que recebem entity_id do LLM.
    Usa SQLAlchemy async session (mesmo pool do projeto).

    Args:
        entity_id: ID da entidade (campaign_id, adset_id, ad_id).
        config_id: ID da config interna (FK para sistema_facebook_ads_config.id).
        entity_type: Tipo da entidade (campaign, adset, ad).

    Returns:
        True se a entidade pertence a conta, False caso contrario.
    """
    entry = MODEL_BY_TYPE.get(entity_type)
    if not entry:
        return False  # entity_type invalido

    model_class, id_column_name = entry
    id_column = getattr(model_class, id_column_name)

    async with async_session_maker() as session:
        result = await session.execute(
            select(model_class.id)
            .where(id_column == entity_id)
            .where(model_class.config_id == config_id)
            .limit(1)
        )
        return result.scalar_one_or_none() is not None
