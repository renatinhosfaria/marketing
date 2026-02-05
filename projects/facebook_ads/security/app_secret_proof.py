"""
Geração de App Secret Proof para requisições autenticadas à Graph API do Facebook.

O App Secret Proof é um HMAC-SHA256 do access token usando o app secret como chave.
Facebook exige este parâmetro para verificar que a requisição vem de um servidor autorizado.
"""

import hashlib
import hmac

from projects.facebook_ads.config import fb_settings


def generate_app_secret_proof(access_token: str) -> str:
    """Generate HMAC-SHA256 App Secret Proof for Facebook Graph API requests."""
    return hmac.new(
        fb_settings.facebook_app_secret.encode("utf-8"),
        access_token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
