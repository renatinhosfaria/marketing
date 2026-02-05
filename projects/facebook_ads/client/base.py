"""
Cliente HTTP base para Facebook Graph API v24.0.
Implementa App Secret Proof, rate limiting proativo e retry com backoff.
"""

import asyncio
import random
from typing import Any, AsyncGenerator, Optional

import httpx

from shared.core.logging import get_logger
from projects.facebook_ads.config import fb_settings
from projects.facebook_ads.security.app_secret_proof import generate_app_secret_proof
from projects.facebook_ads.utils.rate_limiter import rate_limiter

logger = get_logger(__name__)

# Erros que permitem retry
RETRYABLE_ERROR_CODES = {1, 2, 4, 17, 32}
# Erros que NÃO permitem retry (token/permissão)
NON_RETRYABLE_ERROR_CODES = {10, 100, 190, 200, 294}


class FacebookAPIError(Exception):
    """Erro da Facebook Graph API."""

    def __init__(
        self,
        message: str,
        code: int = 0,
        subcode: int = 0,
        fbtrace_id: str = "",
    ):
        super().__init__(message)
        self.code = code
        self.subcode = subcode
        self.fbtrace_id = fbtrace_id


class TokenExpiredError(FacebookAPIError):
    """Token de acesso expirado (erro 190)."""

    pass


class RateLimitError(FacebookAPIError):
    """Rate limit atingido (erros 4, 17, 32)."""

    pass


class FacebookGraphClient:
    """Cliente assíncrono para Facebook Graph API."""

    BASE_URL = "https://graph.facebook.com"
    MAX_RETRIES = 5
    BASE_DELAY = 1.0  # seconds
    REQUEST_TIMEOUT = 60.0
    PAGE_SIZE = 100

    def __init__(self, access_token: str, account_id: str = "default"):
        self.access_token = access_token
        self.account_id = account_id
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def base_url(self) -> str:
        return f"{self.BASE_URL}/{fb_settings.facebook_api_version}"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.REQUEST_TIMEOUT),
                limits=httpx.Limits(
                    max_connections=10, max_keepalive_connections=5
                ),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _add_auth_params(self, params: dict) -> dict:
        """Add access_token and appsecret_proof to params."""
        params["access_token"] = self.access_token
        if fb_settings.facebook_app_secret:
            params["appsecret_proof"] = generate_app_secret_proof(
                self.access_token
            )
        return params

    def _parse_error(self, response_data: dict) -> FacebookAPIError:
        """Parse Facebook API error response."""
        error = response_data.get("error", {})
        code = error.get("code", 0)
        subcode = error.get("error_subcode", 0)
        message = error.get("message", "Erro desconhecido")
        fbtrace_id = error.get("fbtrace_id", "")

        if code == 190:
            return TokenExpiredError(message, code, subcode, fbtrace_id)
        if code in {4, 17, 32}:
            return RateLimitError(message, code, subcode, fbtrace_id)
        return FacebookAPIError(message, code, subcode, fbtrace_id)

    async def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
        retry: bool = True,
    ) -> dict:
        """Make a request to Facebook Graph API with retry and rate limiting."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        params = dict(params or {})
        params = self._add_auth_params(params)

        client = await self._get_client()
        last_error = None

        for attempt in range(self.MAX_RETRIES if retry else 1):
            # Proactive rate limit check
            await rate_limiter.check_and_wait(self.account_id)

            try:
                logger.debug(
                    "Facebook API request",
                    method=method,
                    endpoint=endpoint,
                    attempt=attempt + 1,
                )

                response = await client.request(
                    method, url, params=params, json=data
                )

                # Update rate limiter from headers
                rate_limiter.update_from_headers(
                    dict(response.headers), self.account_id
                )

                response_data = response.json()

                if response.status_code >= 400:
                    error = self._parse_error(response_data)

                    logger.warning(
                        "Facebook API error",
                        endpoint=endpoint,
                        status=response.status_code,
                        error_code=error.code,
                        error_subcode=error.subcode,
                        fbtrace_id=error.fbtrace_id,
                        message=str(error),
                    )

                    # Non-retryable errors
                    if error.code in NON_RETRYABLE_ERROR_CODES:
                        raise error

                    # Retryable errors with backoff
                    if (
                        retry
                        and error.code in RETRYABLE_ERROR_CODES
                        and attempt < self.MAX_RETRIES - 1
                    ):
                        delay = self.BASE_DELAY * (2**attempt) + random.uniform(
                            0, self.BASE_DELAY
                        )
                        logger.info(
                            "Retry após erro",
                            attempt=attempt + 1,
                            delay=delay,
                            error_code=error.code,
                        )
                        await asyncio.sleep(delay)
                        last_error = error
                        continue

                    raise error

                return response_data

            except httpx.TimeoutException as e:
                last_error = FacebookAPIError(f"Timeout: {e}", code=-1)
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.BASE_DELAY * (2**attempt) + random.uniform(
                        0, self.BASE_DELAY
                    )
                    logger.warning(
                        "Timeout - retry", attempt=attempt + 1, delay=delay
                    )
                    await asyncio.sleep(delay)
                    continue
                raise last_error

            except httpx.HTTPError as e:
                last_error = FacebookAPIError(f"HTTP error: {e}", code=-2)
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.BASE_DELAY * (2**attempt)
                    await asyncio.sleep(delay)
                    continue
                raise last_error

        raise last_error or FacebookAPIError("Max retries exceeded")

    async def get(
        self, endpoint: str, params: Optional[dict] = None
    ) -> dict:
        """GET request."""
        return await self.request("GET", endpoint, params=params)

    async def post(
        self,
        endpoint: str,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
    ) -> dict:
        """POST request."""
        return await self.request("POST", endpoint, params=params, data=data)

    async def paginate(
        self,
        endpoint: str,
        params: Optional[dict] = None,
        limit: int = PAGE_SIZE,
    ) -> AsyncGenerator[list[dict], None]:
        """Paginate through Facebook Graph API results using cursor-based pagination."""
        params = dict(params or {})
        params["limit"] = limit

        while True:
            response = await self.get(endpoint, params)
            data = response.get("data", [])

            if data:
                yield data

            # Check for next page
            paging = response.get("paging", {})
            cursors = paging.get("cursors", {})
            next_url = paging.get("next")

            if not next_url or not cursors.get("after"):
                break

            params["after"] = cursors["after"]

    async def get_all(
        self,
        endpoint: str,
        params: Optional[dict] = None,
        limit: int = PAGE_SIZE,
    ) -> list[dict]:
        """Get all results from paginated endpoint."""
        all_results = []
        async for page in self.paginate(endpoint, params, limit):
            all_results.extend(page)
        return all_results
