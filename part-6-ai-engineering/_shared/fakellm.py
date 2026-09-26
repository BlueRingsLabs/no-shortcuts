"""A simulated LLM API for the Part VI labs: realistic enough to engineer against, offline and deterministic.

Why simulate? The labs must run without API keys, network access or money, and they must be reproducible in CI. More
importantly, a fake you control is how you should test LLM code anyway: you can make it slow, rate-limited, flaky or
wrong on purpose, and check that your code copes. Every lesson shows the real SDK calls next to the lab code.

    from fakellm import FakeLLM, VirtualClock
    clock = VirtualClock()
    llm = FakeLLM(brain=my_brain, clock=clock, rpm=60, failure_rate=0.05, seed=0)
    resp = llm.create(messages=[{"role": "user", "content": "hi"}], max_tokens=100)
    resp["content"], resp["usage"], resp["stop_reason"]

- Tokens are counted with the course's BPE tokenizer (35.1), so costs and limits are in real-ish tokens.
- Latency: time to first token grows with the prompt (prefill), then a fixed time per output token (37.2).
- Rate limits: requests per minute and tokens per minute, with HTTP-429-style RateLimitError carrying retry_after.
- Failures: transient server errors (ServerError) and timeouts at configurable rates.
- Prompt caching: a repeated prefix (system prompt + first messages) is billed as cached input tokens.
- The "model" is a brain function: brain(messages, tools, rng) -> {"text": ...} or {"tool_use": {...}}. Labs supply
  brains that behave like models do, including their failure modes.
- Time is virtual: sleeping and latency advance a clock instead of wall time, so a lab can simulate an hour in a second.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "part-5-language-models" / "_shared"))

PRICES = {  # USD per million tokens, illustrative only: check the provider's page, they change often
    "small": {"input": 0.25, "cached_input": 0.025, "output": 1.25},
    "medium": {"input": 3.00, "cached_input": 0.30, "output": 15.00},
    "large": {"input": 15.00, "cached_input": 1.50, "output": 75.00},
}
SPEED = {  # prefill ms per 1k input tokens, ms per output token, fixed overhead ms
    "small": (40, 8, 150), "medium": (120, 25, 250), "large": (300, 60, 400),
}


class APIError(Exception):
    pass


class RateLimitError(APIError):
    def __init__(self, retry_after):
        super().__init__(f"429 rate limited, retry after {retry_after:.1f}s")
        self.retry_after = retry_after


class ServerError(APIError):
    pass


class Timeout(APIError):
    pass


class VirtualClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += max(0.0, seconds)


_TOK = None


def count_tokens(text: str) -> int:
    """Tokens under the course's BPE (trained once and cached in memory). Real APIs expose a token counting endpoint."""
    global _TOK
    if _TOK is None:
        from bpe import BPE
        from corpus import course_text
        _TOK = BPE.train(course_text(), n_merges=1000)
    return len(_TOK.encode(text))


def _text_of(messages, system=""):
    parts = [system]
    for m in messages:
        c = m["content"]
        parts.append(c if isinstance(c, str) else json.dumps(c))
    return "\n".join(parts)


class FakeLLM:
    def __init__(self, brain, clock=None, model="medium", rpm=600, tpm=2_000_000, failure_rate=0.0,
                 timeout_rate=0.0, seed=0, cache_min_tokens=100):
        self.brain, self.clock, self.model = brain, clock or VirtualClock(), model
        self.rpm, self.tpm = rpm, tpm
        self.failure_rate, self.timeout_rate = failure_rate, timeout_rate
        self.rng = random.Random(seed)
        self.window = []                                                    # (time, tokens) of requests in the last minute
        self.cache = {}                                                     # prefix hash -> expiry time
        self.cache_min_tokens = cache_min_tokens
        self.log = []                                                       # every call: for tracing labs (43.2)

    def _check_limits(self, tokens):
        now = self.clock.now()
        self.window = [(t, k) for t, k in self.window if now - t < 60]
        if len(self.window) >= self.rpm or sum(k for _, k in self.window) + tokens > self.tpm:
            oldest = self.window[0][0] if self.window else now
            raise RateLimitError(retry_after=max(0.5, 60 - (now - oldest)))

    def create(self, messages, max_tokens=512, system="", tools=None, temperature=1.0, timeout=60.0,
               cache_prefix_messages=1):
        prompt_text = _text_of(messages, system)
        n_in = count_tokens(prompt_text)
        self._check_limits(n_in + max_tokens)
        self.window.append((self.clock.now(), n_in + max_tokens))
        if self.rng.random() < self.failure_rate:
            self.clock.sleep(0.3)
            self.log.append({"t": self.clock.now(), "ok": False, "error": "server"})
            raise ServerError("500 internal server error (transient)")
        # prompt caching: system prompt + the first messages form a cacheable prefix
        prefix = _text_of(messages[:cache_prefix_messages], system)
        n_prefix = count_tokens(prefix)
        key = hashlib.sha1(prefix.encode()).hexdigest()
        cached = n_prefix if (n_prefix >= self.cache_min_tokens and self.cache.get(key, -1) > self.clock.now()) else 0
        if n_prefix >= self.cache_min_tokens:
            self.cache[key] = self.clock.now() + 300                        # a 5-minute cache lifetime
        out = self.brain(messages, tools or [], self.rng)
        if "tool_use" in out:
            text = json.dumps(out["tool_use"])
            stop = "tool_use"
        else:
            text = out["text"]
            stop = "end_turn"
        n_out = count_tokens(text)
        if n_out > max_tokens:
            text = _TOK.decode(_TOK.encode(text)[:max_tokens])
            n_out, stop = max_tokens, "max_tokens"
        prefill_ms, per_tok_ms, overhead_ms = SPEED[self.model]
        ttft = (overhead_ms + prefill_ms * (n_in - 0.8 * cached) / 1000) / 1000
        total = ttft + per_tok_ms * n_out / 1000
        if self.rng.random() < self.timeout_rate or total > timeout:
            self.clock.sleep(min(total, timeout))
            self.log.append({"t": self.clock.now(), "ok": False, "error": "timeout"})
            raise Timeout(f"no response within {timeout}s")
        self.clock.sleep(total)
        usage = {"input_tokens": n_in - cached, "cached_input_tokens": cached, "output_tokens": n_out}
        p = PRICES[self.model]
        cost = (usage["input_tokens"] * p["input"] + cached * p["cached_input"] + n_out * p["output"]) / 1e6
        resp = {"content": text, "tool_use": out.get("tool_use"), "stop_reason": stop, "usage": usage, "cost": cost,
                "ttft": ttft, "latency": total, "model": self.model}
        self.log.append({"t": self.clock.now(), "ok": True, **{k: resp[k] for k in ("usage", "cost", "latency", "stop_reason")}})
        return resp

    def stream(self, messages, max_tokens=512, system=""):
        """Yields (virtual time, text piece): the first piece after the time to first token, then one per output token."""
        before = self.clock.now()
        resp = self.create(messages, max_tokens=max_tokens, system=system)
        pieces = [_TOK.decode([i]) for i in _TOK.encode(resp["content"])]
        per = (resp["latency"] - resp["ttft"]) / max(1, len(pieces))
        for k, piece in enumerate(pieces):
            yield before + resp["ttft"] + k * per, piece
