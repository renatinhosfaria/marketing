"""
Criptografia de tokens OAuth do Facebook usando AES-256-GCM.

Suporta dois formatos:
- Novo (Python): iv_hex:tag_hex:ciphertext_hex (AES-256-GCM real)
- Legado (Node.js): iv_hex:authTag+ciphertext_hex (AES-128-GCM por bug no Node.js)
"""

import os

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from shared.core.logging import get_logger
from projects.facebook_ads.config import fb_settings

logger = get_logger(__name__)


def _get_encryption_key() -> bytes:
    """Get 32-byte encryption key from hex config. FIXED: uses bytes.fromhex for real 32 bytes."""
    key_hex = fb_settings.facebook_token_encryption_key
    if not key_hex:
        raise ValueError("FACEBOOK_TOKEN_ENCRYPTION_KEY não configurada")
    key_bytes = bytes.fromhex(key_hex)
    if len(key_bytes) < 32:
        raise ValueError(
            f"Chave deve ter 32 bytes (64 hex chars), recebeu {len(key_bytes)} bytes"
        )
    return key_bytes[:32]


def encrypt_token(token: str) -> str:
    """Encrypt token with AES-256-GCM. Returns 'iv_hex:tag_hex:ciphertext_hex'."""
    key = _get_encryption_key()
    iv = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext_with_tag = aesgcm.encrypt(iv, token.encode("utf-8"), None)
    # AESGCM appends 16-byte tag to ciphertext
    ciphertext = ciphertext_with_tag[:-16]
    tag = ciphertext_with_tag[-16:]
    return f"{iv.hex()}:{tag.hex()}:{ciphertext.hex()}"


def decrypt_token(encrypted: str) -> str:
    """Decrypt token. Supports new format 'iv:tag:ciphertext' and legacy Node.js format."""
    parts = encrypted.split(":")
    if len(parts) == 3:
        # New Python format: iv:tag:ciphertext
        iv = bytes.fromhex(parts[0])
        tag = bytes.fromhex(parts[1])
        ciphertext = bytes.fromhex(parts[2])
        key = _get_encryption_key()
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(iv, ciphertext + tag, None)
        return plaintext.decode("utf-8")
    elif len(parts) == 2:
        # Legacy Node.js format: iv_hex:authTag_hex+ciphertext_hex
        return _decrypt_legacy_nodejs(encrypted)
    else:
        raise ValueError(
            f"Formato de token criptografado inválido: {len(parts)} partes"
        )


def _decrypt_legacy_nodejs(encrypted: str) -> str:
    """Decrypt token from Node.js format (AES-256-GCM with string key bug).

    Node.js code uses key.slice(0,32) on a STRING, getting only 16 bytes (AES-128).
    We maintain compatibility by doing the same.
    """
    key_hex = fb_settings.facebook_token_encryption_key
    # Node.js bug: uses string.slice(0,32) which gets 32 chars = 16 bytes of hex decoded
    legacy_key = key_hex[:32].encode("utf-8")[:16]
    # Pad to 16 bytes for AES-128
    if len(legacy_key) < 16:
        legacy_key = legacy_key.ljust(16, b"\0")

    parts = encrypted.split(":")
    if len(parts) != 2:
        raise ValueError("Formato legado inválido")

    iv = bytes.fromhex(parts[0])
    auth_tag_and_cipher = bytes.fromhex(parts[1])

    # Node.js: first 16 bytes = authTag, rest = ciphertext
    auth_tag = auth_tag_and_cipher[:16]
    ciphertext = auth_tag_and_cipher[16:]

    decryptor = Cipher(
        algorithms.AES(legacy_key),
        modes.GCM(iv, auth_tag),
    ).decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()

    logger.warning(
        "Token decriptografado usando formato legado Node.js (AES-128). "
        "Recomenda-se re-encriptar com o novo formato AES-256."
    )
    return plaintext.decode("utf-8")
