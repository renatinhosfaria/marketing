"""Base model com serialização camelCase para o módulo Facebook Ads."""

from pydantic import BaseModel, ConfigDict


def to_camel(string: str) -> str:
    """Converte snake_case → camelCase."""
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


def camel_keys(d: dict) -> dict:
    """Converte chaves de um dict de snake_case para camelCase."""
    return {to_camel(k): v for k, v in d.items()}


class CamelCaseModel(BaseModel):
    """Base model que serializa campos em camelCase."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )
