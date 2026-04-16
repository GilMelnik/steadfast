"""Probe the proxy against the catalog; write only responding model IDs to models.list."""

import json
import os
import urllib.request

from openai import OpenAI

# Fixed settings (no CLI)
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODELS_URL = "https://models.dev/api.json"
MODELS_LIST_FILE = "models.list"
# Catalog has thousands of entries; increase to probe more.
MAX_MODELS = 800


def _catalog_model_ids():
    req = urllib.request.Request(MODELS_URL, headers={"User-Agent": "discover_llm_models/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    ids = set()
    for prov in data.values():
        ids.update((prov.get("models") or {}).keys())
    return sorted(ids)


def _sse_text_from_scratch_style(raw: bytes) -> str:
    """Parse streamed SSE the same way as scratch.py (Anthropic-style deltas)."""
    text = raw.decode("utf-8")
    full = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event_type = None
        data_json = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event_type = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_json = line.removeprefix("data:").strip()
        if not data_json:
            continue
        payload = json.loads(data_json)
        if event_type == "content_block_delta":
            delta = payload.get("delta", {})
            if delta.get("type") == "text_delta":
                full.append(delta["text"])
    return "".join(full)


def main() -> None:
    if not API_KEY:
        raise SystemExit("Set API_KEY in the environment.")

    models = _catalog_model_ids()
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    ok = []
    for model in models[:MAX_MODELS]:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello!"}],
                stream=True,
                max_tokens=1,
            )
            raw = response.response.read()
            reply = _sse_text_from_scratch_style(raw)
            preview = reply[:120] if reply else raw[:120]
            ok.append(model)
            print(f"OK  {model}\t{preview!r}")
        except Exception as e:
            print(f"ERR {model}\t{e!r}")

    with open(MODELS_LIST_FILE, "w", encoding="utf-8") as f:
        f.writelines(m + "\n" for m in ok)
    print(f"Wrote {MODELS_LIST_FILE} ({len(ok)} usable models).")


if __name__ == "__main__":
    main()
