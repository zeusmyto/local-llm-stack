# local-llm-stack

![tests](https://github.com/zeusmyto/local-llm-stack/actions/workflows/ci.yml/badge.svg)

**Deploy a private LLM for a company in one command.** Data stays on the company's own GPU server, and every team can use it through a standard OpenAI-compatible API, with access control.

```
                ┌──────────────┐
 staff  ──────► │  Open WebUI  │ :3000  (chat, login per user)
                └──────┬───────┘
                       │
 apps, scripts, ┌──────▼───────┐        ┌──────────┐
 n8n, LangChain │   gateway    │ :8080 ─►│  Ollama  │  GPU, models on disk
 ─────────────► │ keys · quota │        │ (private)│  not exposed on the host
                │ logs · /metrics       └──────────┘
                └──────────────┘
```

## Why a gateway in front of Ollama?

Ollama is great for running models, but its port has no authentication. Exposing it on a company network means anyone can use (or overload) the GPU. The gateway adds what IT departments ask for:

| Need | What the gateway does |
|---|---|
| Who can use it? | API keys, stored only as SHA-256 hashes, compared in constant time |
| Fair use of one GPU | Per-key rate limit (token bucket) with `Retry-After` |
| Control costs and risk | Model allowlist (`ALLOWED_MODELS`) |
| Privacy | Prompts and answers are **not** logged unless `LOG_PROMPTS=true` |
| Monitoring | JSON logs (key id, model, latency, tokens) and Prometheus `/metrics` |
| Compatibility | Same API as OpenAI: `/v1/chat/completions` (streaming too), `/v1/models` |

## Quick start

Requirements: Docker with the NVIDIA Container Toolkit (or CPU mode, slower).

```bash
cp .env.example .env
python tools/new_key.py            # prints a key + its hash -> put the hash in .env
docker compose up -d
docker compose exec ollama ollama pull llama3.1:8b
```

Without an NVIDIA GPU: `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d`

Test it:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer llm_..." -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b", "messages": [{"role": "user", "content": "Bonjour"}]}'
```

Any OpenAI client works by changing the base URL:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="llm_...")
print(client.chat.completions.create(model="llama3.1:8b",
      messages=[{"role": "user", "content": "Résume ce texte..."}]).choices[0].message.content)
```

## Which model for which GPU?

`tools/vram_estimator.py` estimates the memory a model needs (weights + KV cache + overhead), to size hardware before buying it:

```
$ python tools/vram_estimator.py --vram 12
Models that fit in 12.0 GB (4-bit, 8192 tokens):
  qwen2.5:14b          ~  9.2 GB
  llama3.1:8b          ~  5.2 GB
  mistral:7b           ~  4.8 GB
  llama3.2:3b          ~  2.6 GB
```

## Tests

```bash
pip install -r gateway/requirements.txt pytest
pytest -q
```

The API tests run against a fake Ollama, so they need no GPU. GitHub Actions runs them on every push and builds the Docker image.

## Going further in production

- Put the gateway behind a reverse proxy with TLS (Caddy, Traefik, nginx).
- Several gateway replicas: move rate-limit buckets to Redis.
- Connect `/metrics` to Prometheus + Grafana, and alert on latency or 5xx.
- Plug Open WebUI into the company SSO (OIDC).

---

Built by [Adel Touhami](https://www.linkedin.com/in/adel-touhami-b35a352a) with Claude Code. MIT license.
