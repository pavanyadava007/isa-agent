"""Minimal client for a local Ollama server (no data leaves the machine)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11435")


@dataclass
class Reply:
    text: str
    prompt_tokens: int
    output_tokens: int
    seconds: float


class LLM:
    def __init__(self, model: str, temperature: float = 0.4, num_ctx: int = 16384, max_tokens: int = 1536) -> None:
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.client = httpx.Client(timeout=600, trust_env=False)

    def chat(self, messages: list[dict], seed: int = 0) -> Reply:
        t0 = time.time()
        body = {
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": self.temperature, "seed": seed, "num_ctx": self.num_ctx,
                        "num_predict": self.max_tokens},
        }
        for attempt in range(3):
            try:
                r = self.client.post(f"{OLLAMA_URL}/api/chat", json=body)
                r.raise_for_status()
                d = r.json()
                return Reply(d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0), time.time() - t0)
            except (httpx.HTTPError, KeyError):
                if attempt == 2:
                    raise
                time.sleep(5)
        raise RuntimeError("unreachable")
