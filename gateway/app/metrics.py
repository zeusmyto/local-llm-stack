"""Tiny Prometheus-style counters, no external dependency."""
from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: dict[tuple[str, str, int], int] = defaultdict(int)
        self.latency_sum: dict[str, float] = defaultdict(float)
        self.tokens: dict[tuple[str, str], int] = defaultdict(int)

    def record(self, key_id: str, model: str, status: int, latency_s: float,
               prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        with self._lock:
            self.requests[(key_id, model, status)] += 1
            self.latency_sum[model] += latency_s
            if prompt_tokens:
                self.tokens[(model, "prompt")] += prompt_tokens
            if completion_tokens:
                self.tokens[(model, "completion")] += completion_tokens

    def render(self) -> str:
        lines = [
            "# HELP gateway_requests_total Chat completion requests.",
            "# TYPE gateway_requests_total counter",
        ]
        with self._lock:
            for (kid, model, status), n in sorted(self.requests.items()):
                lines.append(f'gateway_requests_total{{key="{kid}",model="{model}",status="{status}"}} {n}')
            lines += [
                "# HELP gateway_latency_seconds_sum Total upstream latency per model.",
                "# TYPE gateway_latency_seconds_sum counter",
            ]
            for model, total in sorted(self.latency_sum.items()):
                lines.append(f'gateway_latency_seconds_sum{{model="{model}"}} {total:.3f}')
            lines += [
                "# HELP gateway_tokens_total Tokens reported by the model.",
                "# TYPE gateway_tokens_total counter",
            ]
            for (model, kind), n in sorted(self.tokens.items()):
                lines.append(f'gateway_tokens_total{{model="{model}",kind="{kind}"}} {n}')
        return "\n".join(lines) + "\n"
