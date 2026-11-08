"""In-memory token bucket, one bucket per API key.

Good enough for a single gateway instance. For several replicas,
move the buckets to Redis.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    def __init__(self, per_minute: int, clock=time.monotonic):
        if per_minute <= 0:
            raise ValueError("per_minute must be > 0")
        self.capacity = float(per_minute)
        self.refill_per_second = per_minute / 60.0
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = self._buckets[key] = _Bucket(self.capacity, now)
            elapsed = now - bucket.updated
            bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_second)
            bucket.updated = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True
            return False

    def retry_after(self, key: str) -> int:
        """Seconds until the next request would be allowed."""
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None or bucket.tokens >= 1.0:
                return 0
            missing = 1.0 - bucket.tokens
        return max(1, int(missing / self.refill_per_second + 0.999))
