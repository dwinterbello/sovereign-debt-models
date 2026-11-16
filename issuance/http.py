"""Shared HTTP helpers. Refuse HTML captcha pages as if they were data."""

from __future__ import annotations

import time
from pathlib import Path

import requests

AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

_BLOCK_MARKS = (
    "shieldsquare",
    "request unsuccessful",
    "incapsula",
    "captcha",
    "are you a human",
    "bot detection",
    "just a moment",
    "cf-browser-verification",
)


class FetchError(RuntimeError):
    pass


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": AGENT,
        "Accept": "*/*",
        "Accept-Language": "en-GB,en;q=0.9",
    })
    return s


def looks_blocked(payload: bytes) -> bool:
    text = payload[:4000].decode("utf-8", errors="replace").lower()
    return any(mark in text for mark in _BLOCK_MARKS)


def is_workbook(payload: bytes) -> bool:
    return payload.startswith((b"\xd0\xcf\x11\xe0", b"PK\x03\x04"))


def _browser_get(url: str, timeout: int, verify: bool = True):
    """Chrome TLS fingerprint. AFT's Cloudflare answers plain requests with 403.
    ``verify=False`` is for hosts whose intermediate chain breaks OpenSSL (Tesoro)."""
    from curl_cffi import requests as cffi
    return cffi.get(url, impersonate="chrome", timeout=timeout, verify=verify)


def get_bytes(url: str, dest: Path | None = None, timeout: int = 120,
              min_bytes: int = 200, retries: int = 3, impersonate: bool = False,
              verify: bool = True) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            if impersonate:
                resp = _browser_get(url, timeout, verify=verify)
            else:
                resp = session().get(url, timeout=timeout, verify=verify)
            resp.raise_for_status()
            payload = resp.content
            if looks_blocked(payload):
                raise FetchError(f"blocked: {url}")
            if len(payload) < min_bytes:
                raise FetchError(f"too small ({len(payload)} bytes): {url}")
            if dest is not None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(payload)
            return payload
        except Exception as exc:  # noqa: BLE001 — last error is re-raised
            last = exc
            time.sleep(2 ** attempt)
    raise FetchError(f"failed {url}: {last}") from last


def get_text(url: str, timeout: int = 60, impersonate: bool = False,
             verify: bool = True) -> str:
    return get_bytes(url, timeout=timeout, impersonate=impersonate, verify=verify).decode(
        "utf-8", errors="replace")
