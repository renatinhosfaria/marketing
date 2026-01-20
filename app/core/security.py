"""
Segurança e autenticação via API Key e JWT.
"""

import secrets
from typing import Optional, Dict, Any

from fastapi import HTTPException, Security, Depends, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
import jwt

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Bearer token para JWT
bearer_scheme = HTTPBearer(auto_error=False)

# Header para API Key
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header)
) -> str:
    """
    Verifica se a API Key fornecida é válida.

    Args:
        api_key: API Key do header X-API-Key

    Returns:
        API Key validada

    Raises:
        HTTPException: Se a API Key for inválida ou ausente
    """
    if api_key is None:
        logger.warning("Tentativa de acesso sem API Key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key não fornecida",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Comparação segura contra timing attacks
    if not secrets.compare_digest(api_key, settings.ml_api_key):
        logger.warning("Tentativa de acesso com API Key inválida")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API Key inválida",
        )

    return api_key


def generate_api_key(length: int = 32) -> str:
    """
    Gera uma nova API Key segura.

    Args:
        length: Tamanho da chave em bytes (default 32 = 256 bits)

    Returns:
        API Key em formato hexadecimal
    """
    return secrets.token_hex(length)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header),
) -> Dict[str, Any]:
    """
    Obtém o usuário atual a partir do token JWT ou API Key.

    Suporta dois modos de autenticação:
    1. JWT Bearer token (para usuários logados no famachat)
    2. API Key com user_id no header (para chamadas internas)

    Args:
        credentials: Token Bearer JWT
        api_key: API Key do header

    Returns:
        Dicionário com dados do usuário (id, email, role, etc.)

    Raises:
        HTTPException: Se autenticação falhar
    """
    # Tentar JWT primeiro
    if credentials and credentials.credentials:
        try:
            # Decodificar JWT usando o segredo do famachat
            jwt_secret = getattr(settings, 'jwt_secret', None)
            if not jwt_secret:
                jwt_secret = settings.ml_api_key  # Fallback

            payload = jwt.decode(
                credentials.credentials,
                jwt_secret,
                algorithms=["HS256"],
                options={
                    "verify_exp": True,
                    "verify_aud": False,  # Ignorar verificação de audience
                }
            )

            return {
                "id": payload.get("id") or payload.get("userId") or payload.get("sub"),
                "email": payload.get("email"),
                "role": payload.get("role", "user"),
                "name": payload.get("name"),
            }

        except jwt.ExpiredSignatureError:
            logger.warning("Token JWT expirado")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expirado",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token JWT inválido: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # Tentar API Key como fallback
    if api_key:
        if secrets.compare_digest(api_key, settings.ml_api_key):
            # API Key válida, retornar usuário de sistema
            return {
                "id": 0,
                "email": "system@famachat.ml",
                "role": "system",
                "name": "Sistema ML",
            }

    # Nenhuma autenticação válida
    logger.warning("Tentativa de acesso sem autenticação válida")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Autenticação necessária",
        headers={"WWW-Authenticate": "Bearer"},
    )
