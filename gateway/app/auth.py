"""API key checks. Keys are compared as SHA-256 hashes in constant time."""
from __future__ import annotations

import hashlib
import hmac
import secrets


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def new_key(prefix: str = "llm") -> str:
    """Return a new random API key, e.g. 'llm_3f9a...'."""
    return f"{prefix}_{secrets.token_urlsafe(32)}"


def key_id(key_hash: str) -> str:
    """Short, non-secret identifier for logs and metrics."""
    return key_hash[:8]


def extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def check_key(key: str | None, allowed_hashes: list[str]) -> str | None:
    """Return the matching hash if the key is allowed, otherwise None."""
    if not key or not allowed_hashes:
        return None
    candidate = hash_key(key)
    match = None
    for allowed in allowed_hashes:
        # compare every entry so timing does not reveal which one matched
        if hmac.compare_digest(candidate, allowed):
            match = allowed
    return match
