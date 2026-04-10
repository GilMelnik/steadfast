"""Fetch and display all available models from the configured LLM endpoint."""

from __future__ import annotations

import os
from typing import List

from openai import OpenAI


def get_client() -> OpenAI:
    api_key = os.getenv("API_KEY") or os.getenv("LSP_API_KEY") or ""
    base_url = os.getenv("BASE_URL") or os.getenv("LSP_API_BASE") or ""
    return OpenAI(api_key=api_key, base_url=base_url)


def fetch_available_models(client: OpenAI) -> List[str]:
    models = client.models.list()
    return sorted(model.id for model in models.data)


def print_models(model_ids: List[str]) -> None:
    print(f"Available models ({len(model_ids)}):")
    for model_id in model_ids:
        print(f"  - {model_id}")


if __name__ == "__main__":
    client = get_client()
    model_ids = fetch_available_models(client)
    print_models(model_ids)

