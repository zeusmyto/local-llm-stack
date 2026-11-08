"""Which local model fits on which GPU?

Rough estimate of the VRAM a model needs, so you can pick a model
for a client's hardware (or the hardware for a client's model).

    python tools/vram_estimator.py --vram 12
    python tools/vram_estimator.py --model llama3.1:8b --quant 4 --context 8192

Rule of thumb used here:
    weights  = parameters x bits / 8
    kv cache = 2 x layers x context x kv_heads x head_dim x 2 bytes (fp16)
    total    = (weights + kv cache) x 1.10  (runtime overhead)
These are estimates: real usage depends on the runtime and settings.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_b: float      # billions of parameters
    layers: int
    kv_heads: int
    head_dim: int


# Public architecture figures for common open-weight models.
CATALOG: dict[str, ModelSpec] = {
    "llama3.2:3b": ModelSpec("llama3.2:3b", 3.2, 28, 8, 128),
    "mistral:7b": ModelSpec("mistral:7b", 7.2, 32, 8, 128),
    "llama3.1:8b": ModelSpec("llama3.1:8b", 8.0, 32, 8, 128),
    "qwen2.5:14b": ModelSpec("qwen2.5:14b", 14.7, 48, 8, 128),
    "mistral-small:24b": ModelSpec("mistral-small:24b", 23.6, 40, 8, 128),
    "qwen2.5:32b": ModelSpec("qwen2.5:32b", 32.8, 64, 8, 128),
    "llama3.3:70b": ModelSpec("llama3.3:70b", 70.6, 80, 8, 128),
}

OVERHEAD = 1.10


def weights_gb(params_b: float, bits: int) -> float:
    return params_b * 1e9 * bits / 8 / 1024**3


def kv_cache_gb(spec: ModelSpec, context: int) -> float:
    bytes_total = 2 * spec.layers * context * spec.kv_heads * spec.head_dim * 2
    return bytes_total / 1024**3


def estimate_gb(spec: ModelSpec, bits: int = 4, context: int = 8192) -> float:
    return round((weights_gb(spec.params_b, bits) + kv_cache_gb(spec, context)) * OVERHEAD, 1)


def fitting_models(vram_gb: float, bits: int = 4, context: int = 8192) -> list[tuple[str, float]]:
    """Models that fit, largest first."""
    fits = [(name, estimate_gb(spec, bits, context)) for name, spec in CATALOG.items()]
    fits = [f for f in fits if f[1] <= vram_gb]
    return sorted(fits, key=lambda f: CATALOG[f[0]].params_b, reverse=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vram", type=float, help="available GPU memory in GB")
    p.add_argument("--model", choices=sorted(CATALOG), help="estimate one model")
    p.add_argument("--quant", type=int, default=4, choices=[4, 5, 8, 16], help="bits per weight (default 4)")
    p.add_argument("--context", type=int, default=8192, help="context length in tokens (default 8192)")
    args = p.parse_args(argv)

    if args.model:
        spec = CATALOG[args.model]
        print(f"{args.model} at {args.quant}-bit, {args.context} tokens of context:")
        print(f"  weights  {weights_gb(spec.params_b, args.quant):6.1f} GB")
        print(f"  kv cache {kv_cache_gb(spec, args.context):6.1f} GB")
        print(f"  total   ~{estimate_gb(spec, args.quant, args.context):6.1f} GB (with {int((OVERHEAD - 1) * 100)}% overhead)")
        return 0
    if args.vram:
        fits = fitting_models(args.vram, args.quant, args.context)
        if not fits:
            print(f"No catalog model fits in {args.vram} GB at {args.quant}-bit. Try --quant 4 or a smaller context.")
            return 1
        print(f"Models that fit in {args.vram} GB ({args.quant}-bit, {args.context} tokens):")
        for name, gb in fits:
            print(f"  {name:<20} ~{gb:5.1f} GB")
        return 0
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
