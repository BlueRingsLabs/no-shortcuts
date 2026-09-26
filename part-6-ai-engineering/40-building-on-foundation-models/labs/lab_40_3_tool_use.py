# %% [markdown]
# # Lab 40.3: Tool use and function calling
#
# A support assistant with three tools (look up an order, refund it, email the customer), against the simulated API.
# The "model" is a scripted brain that behaves like models do when calling tools, including their mistakes: a wrong
# order ID now and then, an amount in the wrong type, a loop that doesn't know when to stop. What the lab measures is
# the code around it.
# 1. The tool loop, traced.
# 2. Validating arguments before running anything, returning errors to the model, and scoping tools to the user.
# 3. Side effects and retries: a payment service that sometimes times out *after* charging. Idempotency keys.
# 4. Budgets and approvals: step limits, and a human in the loop above a threshold.

# %%
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from fakellm import FakeLLM, VirtualClock  # noqa: E402

TOOLS = [
    {"name": "lookup_order", "description": "Get an order by ID: items, amount paid in cents, status, customer ID.",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string", "pattern": "^A[0-9]{4}$"}},
                      "required": ["order_id"]}},
    {"name": "refund", "description": "Refund an order, fully or partly. Irreversible. Amount in cents, at most the amount paid.",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "amount_cents": {"type": "integer", "minimum": 1}},
                      "required": ["order_id", "amount_cents"]}},
    {"name": "send_email", "description": "Email a customer. Use once, to confirm the outcome.",
     "input_schema": {"type": "object", "properties": {"customer_id": {"type": "string"}, "body": {"type": "string"}},
                      "required": ["customer_id", "body"]}},
]


class LookupArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    order_id: str = Field(pattern=r"^A[0-9]{4}$")


class RefundArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    order_id: str = Field(pattern=r"^A[0-9]{4}$")
    amount_cents: int = Field(gt=0)


class EmailArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    customer_id: str
    body: str = Field(min_length=1, max_length=2000)


ARGS = {"lookup_order": LookupArgs, "refund": RefundArgs, "send_email": EmailArgs}

# %% [markdown]
# ## The world: an order database, a flaky payment service, a mail server

# %%
class World:
    def __init__(self, seed, timeout_after=0.0, timeout_before=0.0):
        r = random.Random(seed)
        self.orders = {f"A{1000 + i}": {"customer": f"C{5000 + i}", "amount_cents": r.choice([1999, 4999, 12900, 34900]),
                                        "status": "delivered"} for i in range(2000)}
        self.refunds, self.emails = [], []
        self.seen_keys = {}
        self.r, self.timeout_after, self.timeout_before = r, timeout_after, timeout_before

    def lookup_order(self, order_id):
        if order_id not in self.orders:
            raise KeyError(f"order {order_id} not found")
        return {"order_id": order_id, **self.orders[order_id]}

    def refund(self, order_id, amount_cents, idempotency_key=None):
        """A payment API. Like real ones, it can time out before or after doing the work, and supports idempotency keys."""
        if self.r.random() < self.timeout_before:
            raise TimeoutError("payment service timeout")
        if idempotency_key is not None and idempotency_key in self.seen_keys:
            return self.seen_keys[idempotency_key]                            # same key: return the original result, do nothing
        amount_cents = int(float(str(amount_cents).lstrip("$")))              # glue code 'being helpful' with odd inputs
        paid = self.orders[order_id]["amount_cents"]
        refunded = sum(a for o, a in self.refunds if o == order_id)
        if amount_cents + refunded > paid:
            raise ValueError(f"refund would exceed amount paid ({paid - refunded} cents refundable)")
        self.refunds.append((order_id, amount_cents))
        result = {"refund_id": f"R{len(self.refunds)}", "order_id": order_id, "amount_cents": amount_cents}
        if idempotency_key is not None:
            self.seen_keys[idempotency_key] = result
        if self.r.random() < self.timeout_after:
            raise TimeoutError("payment service timeout")                    # the refund happened; the caller can't know
        return result

    def send_email(self, customer_id, body):
        self.emails.append((customer_id, body))
        return {"sent": True}


# %% [markdown]
# ## The model: a scripted brain with realistic mistakes

# %%
def make_brain(p_bad_id=0.1, p_bad_type=0.1, p_loop=0.0):
    def brain(messages, tools, rng):
        user = messages[0]["content"]
        oid = re.search(r"A\d{4}", user).group(0)
        results = [json.loads(m["content"][0]["content"]) for m in messages if m["role"] == "user" and isinstance(m["content"], list)]
        last = results[-1] if results else None
        calls = [json.loads(m["content"]) for m in messages if m["role"] == "assistant" and m["content"].startswith("{")]
        names = [c["name"] for c in calls]
        if rng.random() < p_loop and len(calls) >= 1:                          # the stuck agent: looks the order up again
            return {"tool_use": {"id": f"t{len(calls)}", "name": "lookup_order", "input": {"order_id": oid}}}
        order = next((r["result"] for r in results if r.get("ok") and "customer" in r.get("result", {})), None)
        if order is None:
            bad = not names and rng.random() < p_bad_id                        # first attempt sometimes garbles the ID
            use = oid[:-1] + str((int(oid[-1]) + 1) % 10) if bad else oid
            return {"tool_use": {"id": f"t{len(calls)}", "name": "lookup_order", "input": {"order_id": use}}}
        refunded = any(r.get("ok") and "refund_id" in r.get("result", {}) for r in results)
        if not refunded and not (last and last.get("needs_approval")):
            if last and not last.get("ok") and "refund" in names[-1:] and "exceed" in last.get("error", ""):
                return {"text": "I couldn't refund this order: it has already been refunded."}
            wrong_type = names.count("refund") == 0 and rng.random() < p_bad_type
            cents = 599 if "shipping" in user else order["amount_cents"]
            amount = f"${cents / 100:.2f}" if wrong_type else cents
            return {"tool_use": {"id": f"t{len(calls)}", "name": "refund", "input": {"order_id": order["order_id"], "amount_cents": amount}}}
        if last and last.get("needs_approval"):
            return {"text": "Your refund needs a manager's approval; you'll hear from us within a day."}
        if "send_email" not in names:
            return {"tool_use": {"id": f"t{len(calls)}", "name": "send_email",
                                 "input": {"customer_id": order["customer"], "body": "Your refund has been issued."}}}
        return {"text": "Done: the order is refunded and the customer has been emailed."}
    return brain


# %% [markdown]
# ## 1. The tool loop

# %%
def run_agent(llm, world, request, customer, validate=True, scoped=True, idempotent=True, retry_tools=True, max_steps=10,
              approval_over=None, conv_id="c0", trace=False):
    """The loop every tool-using application runs: call the model; if it asks for a tool, check and run it, append the
    result, repeat; stop at a final answer or at the step budget. Returns (final text, steps, status).
    `customer` comes from authentication, never from the model: with scoped=True, tools only see that customer's orders."""
    messages = [{"role": "user", "content": request}]
    for step in range(max_steps):
        resp = llm.create(messages, tools=TOOLS, max_tokens=300)
        if resp["stop_reason"] != "tool_use":
            if trace:
                print(f"  [{step}] model: {resp['content']}")
            return resp["content"], step + 1, "done"
        call = resp["tool_use"]
        messages.append({"role": "assistant", "content": json.dumps(call)})
        name, args = call["name"], call["input"]
        try:
            if validate:
                ARGS[name].model_validate(args)                               # before anything runs
            if scoped and "order_id" in args and world.orders.get(args["order_id"], {}).get("customer") != customer:
                raise KeyError(f"order {args['order_id']} not found")        # same answer as a missing order: no oracle
            if name == "refund" and approval_over is not None and args["amount_cents"] > approval_over:
                result = {"ok": False, "needs_approval": True, "error": "queued for human approval"}
            else:
                fn = getattr(world, name)
                kwargs = dict(args)
                if name == "refund" and idempotent:
                    kwargs["idempotency_key"] = f"{conv_id}:{args['order_id']}:{args['amount_cents']}"
                attempts = 3 if retry_tools else 1
                for a in range(attempts):
                    try:
                        result = {"ok": True, "result": fn(**kwargs)}
                        break
                    except TimeoutError:
                        if a == attempts - 1:
                            raise
        except ValidationError as e:
            result = {"ok": False, "error": "invalid arguments: " + "; ".join(err["msg"] for err in e.errors())}
        except (KeyError, ValueError, TimeoutError) as e:
            result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        except Exception as e:                                                # what unvalidated arguments do downstream
            result = {"ok": False, "error": f"crash: {type(e).__name__}: {e}"}
        if trace:
            print(f"  [{step}] model calls {name}({json.dumps(args)}) -> {json.dumps(result)[:110]}")
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"],
                                                      "content": json.dumps(result)}]})
    return None, max_steps, "step budget exhausted"


world = World(seed=0)
llm = FakeLLM(brain=make_brain(p_bad_id=1.0, p_bad_type=1.0), clock=VirtualClock(), model="small", seed=3)
print("one conversation, with a model that garbles the order ID and the amount type on its first tries:")
text, steps, status = run_agent(llm, world, "Order A1042 arrived broken. I want my money back.", customer="C5042", trace=True)
print(f"status: {status} after {steps} model calls; refunds issued: {world.refunds}; emails: {len(world.emails)}")
assert status == "done" and world.refunds == [("A1042", world.orders["A1042"]["amount_cents"])]

# %% [markdown]
# ## 2. Validate arguments, and scope what tools can touch

# %%
def batch(n, seed, **kw):
    world = World(seed=seed, timeout_after=kw.pop("timeout_after", 0.0), timeout_before=kw.pop("timeout_before", 0.0))
    brain_kw = {k: kw.pop(k) for k in ("p_bad_id", "p_bad_type", "p_loop") if k in kw}
    llm = FakeLLM(brain=make_brain(**brain_kw), clock=VirtualClock(), model="small", seed=seed)
    shipping = kw.pop("shipping", False)
    targets = random.Random(seed).sample(sorted(world.orders), n)
    request = "Order {} arrived late. Please refund the shipping." if shipping else "Order {} arrived broken. I want my money back."
    out = [run_agent(llm, world, request.format(o), world.orders[o]["customer"], conv_id=f"c{i}", **kw)
           for i, o in enumerate(targets)]
    others = sum(a for o, a in world.refunds if o not in set(targets))
    per_order = {o: sum(a for oo, a in world.refunds if oo == o) for o in targets}
    due = {o: 599 if shipping else world.orders[o]["amount_cents"] for o in targets}
    return {"correct": sum(per_order[o] == due[o] for o in targets), "double": sum(per_order[o] > due[o] for o in targets),
            "wrong amount": sum(0 < per_order[o] < due[o] for o in targets), "missing": sum(per_order[o] == 0 for o in targets),
            "other orders": others, "steps": np.mean([s for _, s, _ in out]),
            "exhausted": sum(st != "done" for _, _, st in out), "out": out}


print("\n300 conversations; the model garbles the order ID on 10% of first lookups and sends the amount as a string")
print("like '$49.99' on 10% of first refunds:")
print(f"  {'checks in the tool layer':38s} {'correct':>8s} {'wrong amount':>13s} {'other customers refunded':>25s}")
val = {}
for name, kw in (("none", dict(validate=False, scoped=False)), ("argument validation", dict(validate=True, scoped=False)),
                 ("validation + scoped to the customer", dict(validate=True, scoped=True))):
    r = batch(300, seed=1, **kw)
    val[name] = r
    print(f"  {name:38s} {r['correct']:8d} {r['wrong amount']:13d} {'$' + format(r['other orders'] / 100, ',.2f'):>25s}")
print("without validation, '$49.99' reached glue code that 'helpfully' converted it and refunded 49 cents: the call")
print("succeeded, the model reported success. A strict schema check returns a precise error ('Input should be a valid")
print("integer') and the model fixes its own call. But a garbled ID is often another customer's valid ID: the lookup")
print("succeeds, and the refund goes to a stranger. No schema catches that. Authorization does: tools act only within")
print("the authenticated user's scope, enforced in code, whatever the model asks for.")
assert val["validation + scoped to the customer"]["correct"] == 300 and val["none"]["wrong amount"] > 0
assert val["argument validation"]["other orders"] > 0 and val["validation + scoped to the customer"]["other orders"] == 0

# %% [markdown]
# ## 3. Side effects and retries
#
# Customers ask for the shipping fee back (a partial refund, so the payment service's "not more than was paid" check
# doesn't catch a duplicate). The payment service times out 10% of the time *after* refunding, and 5% of the time
# before. Three clients: never retry tool calls; retry them blindly; retry them with an idempotency key.

# %%
print("\n1000 conversations, payment service timing out 5% before and 10% after doing the work:")
print(f"  {'client':32s} {'correct':>8s} {'refunded twice':>15s} {'not refunded':>13s}")
side = {}
for name, kw in (("no retries", dict(retry_tools=False, idempotent=False)),
                 ("retries, no idempotency key", dict(retry_tools=True, idempotent=False)),
                 ("retries + idempotency key", dict(retry_tools=True, idempotent=True))):
    r = batch(1000, seed=2, timeout_after=0.10, timeout_before=0.05, p_bad_id=0.0, p_bad_type=0.0, shipping=True, **kw)
    side[name] = r
    print(f"  {name:32s} {r['correct']:8d} {r['double']:15d} {r['missing']:13d}")
print("a timeout is an unknown outcome, not a failure. Not retrying in your code changed nothing: the model saw an error")
print("and called the tool again, so 'maybe refunded' became 'refunded twice' either way. Only the idempotency key helps:")
print("the service recognizes the repeat and returns the original result. The key must come from your code, derived from")
print("the conversation and the operation, never from the model, because the model's retry is the one you can't stop.")
assert side["retries + idempotency key"]["double"] == 0 and side["retries, no idempotency key"]["double"] > 0
assert side["retries + idempotency key"]["correct"] > side["no retries"]["correct"]

# %% [markdown]
# ## 4. Budgets and approvals

# %%
r_loop = batch(300, seed=3, p_bad_id=0.0, p_bad_type=0.0, p_loop=0.3, max_steps=8)
print(f"\na model that gets stuck re-reading the order 30% of the time, step budget 8: {r_loop['exhausted']} of 300 "
      f"conversations hit the budget and were handed off; mean {r_loop['steps']:.1f} model calls")
r_appr = batch(300, seed=4, p_bad_id=0.0, p_bad_type=0.0, approval_over=20000)
queued = sum("approval" in (t or "") for t, _, _ in r_appr["out"])
print(f"refunds above $200 require a human: {queued} of 300 queued for approval, {r_appr['correct']} refunded automatically")
print("every loop needs a budget (steps, tokens, money, time) and a defined thing to do when it runs out. Irreversible or")
print("expensive actions get a human, enforced in the tool layer where the model can't talk its way around it.")
assert r_loop["exhausted"] > 0 and queued > 0 and r_appr["correct"] + queued == 300

print("\nAll checks passed.")
