# %% [markdown]
# # Lab 40.2: Prompting as programming, and structured outputs
#
# 1. What models actually return when you ask for JSON: twelve realistic outputs, three parsers, two validators.
# 2. Validate and retry, against the simulated API: and why retries help less than the arithmetic promises.
# 3. Prompt formatting bugs on a real (tiny) model: the course's fine-tuned GPT from 38.1, with its chat template
#    applied correctly, with a trailing space removed, and with the wrong template.
# 4. Asking the same model for JSON directly vs letting it answer and building the JSON in code.

# %%
import ast
import json
import re
import random
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "part-5-language-models" / "_shared"))
from fakellm import FakeLLM, VirtualClock  # noqa: E402

# %% [markdown]
# ## 1. Parsing is not validating
#
# The task: triage a support ticket into {"priority", "category", "refund_requested", "summary"}. Below, twelve
# outputs of the kinds models produce when a prompt merely *asks* for JSON. Every one of them has been seen in the wild.

# %%
class Triage(BaseModel):
    priority: Literal["low", "medium", "high"]
    category: str
    refund_requested: bool
    summary: str


class StrictTriage(Triage):
    model_config = ConfigDict(strict=True)


GOOD = '{"priority": "high", "category": "billing", "refund_requested": true, "summary": "Charged twice this month."}'
EXAMPLE = '{"priority": "low", "category": "other", "refund_requested": false, "summary": "Example."}'
OUTPUTS = {
    "clean": GOOD,
    "markdown fence": f"```json\n{GOOD}\n```",
    "chatty preamble": f"Sure! Here is the JSON you asked for:\n\n{GOOD}\n\nLet me know if you need anything else.",
    "trailing comma": GOOD[:-1] + ",}",
    "python dict": GOOD.replace('"', "'").replace("true", "True"),
    "comment inside": GOOD.replace('"billing",', '"billing",  // could also be account\n'),
    "truncated (max_tokens)": '{"priority": "high", "category": "billing", "refund_requested": true, "summary": "Charged tw',
    "invented enum value": GOOD.replace('"high"', '"urgent"'),
    "bool as a string": GOOD.replace("true", '"yes"'),
    "missing field": '{"priority": "high", "category": "billing", "summary": "Charged twice this month."}',
    "echoed the example first": f"Following the format of {EXAMPLE}, the answer is {GOOD}",
    "extra field": GOOD[:-1] + ', "confidence": 0.9}',
}


def parse_strict(raw):
    return json.loads(raw)


def first_object(raw):
    """The first balanced {...} in the text, respecting strings."""
    start = raw.find("{")
    if start < 0:
        raise ValueError("no object")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(raw[start:], start):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    raise ValueError("unbalanced")


def parse_extract(raw):
    return json.loads(first_object(raw))


def parse_lenient(raw):
    """What 'JSON repair' helpers do: strip comments and trailing commas, accept Python literals, close what's open."""
    s = raw[raw.find("{"):] if "{" in raw else raw
    try:
        s = first_object(s)
    except ValueError:                                                     # truncated: close the string and the object
        s = s + ('"' if s.count('"') % 2 else "") + "}"
    s = re.sub(r"//[^\n]*", "", s)
    s = re.sub(r",\s*}", "}", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)


def outcome(raw, parser, schema):
    try:
        obj = parser(raw)
    except Exception:
        return "parse error"
    try:
        t = schema.model_validate(obj)
    except ValidationError:
        return "invalid"
    return "ok" if t.model_dump() == Triage.model_validate(json.loads(GOOD)).model_dump() else "ok, WRONG"


print(f"{'output':26s} {'json.loads':12s} {'extract':12s} {'lenient':12s} {'lenient+strict':14s}")
table = {}
for name, raw in OUTPUTS.items():
    row = [outcome(raw, parse_strict, Triage), outcome(raw, parse_extract, Triage), outcome(raw, parse_lenient, Triage),
           outcome(raw, parse_lenient, StrictTriage)]
    table[name] = row
    print(f"{name:26s} " + " ".join(f"{c:12s}" for c in row))
cols = list(zip(*table.values()))
print("\n'ok' = parsed, validated and equal to the intended answer; 'ok, WRONG' = passed validation with the wrong content")
for label, col in zip(("json.loads", "extract", "lenient", "lenient + strict"), cols):
    print(f"  {label:17s} ok {col.count('ok'):2d}   silently wrong {col.count('ok, WRONG')}")
print("the lenient parser rescues the most outputs, and also turns a truncated answer into a 'valid' object with half")
print("a summary; pydantic's default (lax) mode turned the string 'yes' into True. Check stop_reason before parsing,")
print("validate strictly, and prefer an API mode that constrains the output to the schema (constrained decoding, 37.1).")
assert table["truncated (max_tokens)"][2] == "ok, WRONG" and table["bool as a string"][2] == "ok"
assert table["bool as a string"][3] == "invalid" and table["echoed the example first"][1] == "ok, WRONG"

# %% [markdown]
# ## 2. Validate and retry, and when retrying doesn't help
#
# The standard loop: call, check stop_reason, parse, validate; on failure, send the error back and try again, a bounded
# number of times. Against the simulated API, with a "model" that returns a bad output with some probability. Two
# worlds with the same 10% average failure rate: in one, every ticket is equally likely to fail; in the other, a few
# tickets are hard (the model fails them almost every time) and most are easy.

# %%
BAD = [v for k, v in OUTPUTS.items() if k in ("truncated (max_tokens)", "invented enum value", "missing field", "python dict")]


def make_brain(p_fail_per_ticket):
    def brain(messages, tools, rng):
        tid = messages[0]["ticket_id"]
        return {"text": rng.choice(BAD) if rng.random() < p_fail_per_ticket[tid] else GOOD}
    return brain


def structured_call(llm, messages, schema, max_attempts=3):
    """Returns (object or None, attempts, cost). Errors go back to the model as a user message."""
    msgs, cost = list(messages), 0.0
    for attempt in range(1, max_attempts + 1):
        r = llm.create(msgs, max_tokens=200)
        cost += r["cost"]
        try:
            if r["stop_reason"] == "max_tokens":
                raise ValueError("output truncated")
            return schema.model_validate(parse_extract(r["content"])), attempt, cost
        except Exception as e:                                             # parse, validation or truncation
            msgs = msgs + [{"role": "assistant", "content": r["content"]},
                           {"role": "user", "content": f"That was not valid: {str(e)[:200]}. Reply with only the corrected JSON."}]
    return None, max_attempts, cost


N = 10_000
rs = np.random.default_rng(0)
worlds = {"every ticket 10%": np.full(N, 0.10),
          "hard tickets (Beta(0.1, 0.9), mean 10%)": rs.beta(0.1, 0.9, N)}
print(f"\n{N} tickets, up to 3 attempts each:")
print(f"  {'world':40s} {'mean p':>7s} {'failed after 3':>15s} {'if independent':>15s} {'calls/ticket':>13s}")
res = {}
for name, p in worlds.items():
    llm = FakeLLM(brain=make_brain(p), clock=VirtualClock(), model="small", seed=1)
    out = [structured_call(llm, [{"role": "user", "content": "Triage this ticket: charged twice.", "ticket_id": i}],
                           StrictTriage) for i in range(N)]
    failed = np.mean([o[0] is None for o in out])
    res[name] = failed
    print(f"  {name:40s} {p.mean():7.3f} {failed:15.4f} {p.mean() ** 3:15.4f} {np.mean([o[1] for o in out]):13.2f}")
print("retries multiply failure rates only when failures are independent. When they concentrate on hard inputs, the")
print("same retry budget leaves ~40 times more failures: those inputs need a better prompt, a better model or a human,")
print("not a fourth attempt. Log which inputs fail (43.2), and check how concentrated your failures are.")
assert res["every ticket 10%"] < 0.003 and res["hard tickets (Beta(0.1, 0.9), mean 10%)"] > 0.02

# %% [markdown]
# ## 3. Prompt formatting bugs, on a real model
#
# The course's tiny GPT, fine-tuned on course questions with the chat template "<|user|> Q\n<|assistant|> " (38.1).
# Nothing below changes the model; only the string we feed it.

# %%
from instructions import END, dataset, format_chat, load_or_train_sft  # noqa: E402

torch.set_num_threads(4)
model, tok = load_or_train_sft(verbose=False)
end_ids = tok.encode(" " + END)


@torch.no_grad()
def complete(prompt, max_new=30, stop=None):
    ids = tok.encode(prompt)
    start = len(ids)
    logits, caches = model(torch.tensor([ids]), [None] * len(model.blocks), 0)
    for _ in range(max_new):
        t = int(logits[0, -1].argmax())
        ids.append(t)
        text = tok.decode(ids[start:])
        if ids[-len(end_ids):] == end_ids:
            return tok.decode(ids[start:-len(end_ids)]).strip()
        if stop and stop in text:
            return text.split(stop)[0]
        if len(ids) >= model.ctx:
            break
        logits, caches = model(torch.tensor([[t]]), caches, len(ids) - 1)
    return tok.decode(ids[start:]).strip()


items = [x for x in dataset("train") if x[2] == "lesson_title"]
random.Random(0).shuffle(items)
items = items[:120]
variants = {
    "correct template": lambda q: format_chat(q),
    "trailing space stripped": lambda q: format_chat(q).rstrip(),
    "a friendly instruction added": lambda q: format_chat("Please answer briefly. " + q),
    "wrong template (User:/Assistant:)": lambda q: f"User: {q}\nAssistant: ",
}
print(f"\nthe same 120 training questions (lesson titles), exact-match accuracy:")
acc = {}
for name, f in variants.items():
    acc[name] = np.mean([complete(f(q)) == a for q, a, _ in items])
    print(f"  {name:36s} {acc[name]:.3f}")
print(f"tokens at the end of the prompt, correct vs stripped: {[tok.decode([i]) for i in tok.encode(format_chat('x'))[-3:]]} vs "
      f"{[tok.decode([i]) for i in tok.encode(format_chat('x').rstrip())[-3:]]}")
print("the model learned to continue after one exact token sequence. A stripped space or a hand-written template is a")
print("different input. With hosted APIs the provider applies the template; when you self-host (45.2) or build prompts")
print("from raw strings, use the model's own template (tokenizer.apply_chat_template) and test it.")
assert acc["correct template"] > acc["wrong template (User:/Assistant:)"] + 0.2

# %% [markdown]
# ## 4. JSON directly, or JSON in code?
#
# Same questions. (a) Let the model answer as it was trained to, and build {"title": ...} in code. (b) Prefill the
# assistant turn with '{"title": "' so the model writes the value inside JSON, stopping at the closing quote: the
# cheapest form of constraining the output format.

# %%
direct = [complete(format_chat(q)) for q, a, _ in items]
in_json = [complete(format_chat(q) + '{"title": "', stop='"') for q, a, _ in items]
acc_code = np.mean([d == a for d, (_, a, _) in zip(direct, items)])
acc_json = np.mean([j.strip() == a for j, (_, a, _) in zip(in_json, items)])
print(f"\nanswer, then JSON built in code: {acc_code:.3f};  answer written inside a JSON prefill: {acc_json:.3f}")
print("examples inside JSON:", [j for j in in_json[:3]], "truth:", [a for _, a, _ in items[:3]])
print("the model never saw JSON around its answers, and the format alone knocked it off. Large models are more robust")
print("to this than a 1M-parameter one, and not immune (Tam et al., 2024). When a format costs accuracy: answer first")
print("and wrap in code, or put a free-text field before the answer fields, or fine-tune on the format.")
assert acc_code >= acc_json

print("\nAll checks passed.")
