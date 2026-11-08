"""Generate an API key for the gateway.

    python tools/new_key.py

Give the key to the user or application, and put only the hash
in API_KEY_HASHES (in .env). The gateway never stores plain keys.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))

from app.auth import hash_key, new_key  # noqa: E402

if __name__ == "__main__":
    key = new_key()
    print("API key (give to the client, shown once):")
    print(f"  {key}")
    print("Hash (add to API_KEY_HASHES in .env):")
    print(f"  {hash_key(key)}")
