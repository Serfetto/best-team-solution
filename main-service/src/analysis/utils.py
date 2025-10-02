from typing import Any
from urllib.parse import urljoin
import aiohttp
from fastapi import HTTPException, status
from src.configs.config import settings


async def _request_service(method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
    try:
        async with aiohttp.ClientSession() as session:
            request_kwargs: dict[str, Any] = {"json": payload} if payload is not None else {}
            async with session.request(method.upper(), url, **request_kwargs) as response:
                try:
                    body = await response.json()
                except aiohttp.ContentTypeError:
                    body = await response.text()

                if response.status >= 400:
                    detail = body.get("message") if isinstance(body, dict) else body
                    raise HTTPException(status_code=response.status, detail=detail)

                return body
    except aiohttp.ClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Service request failed: {exc}",
        ) from exc

async def _post_to_service(url: str, payload: dict[str, Any]) -> Any:
    return await _request_service("post", url, payload)


async def _put_to_service(url: str, payload: dict[str, Any]) -> Any:
    return await _request_service("put", url, payload)


async def _get_from_service(url: str) -> Any:
    return await _request_service("get", url)


