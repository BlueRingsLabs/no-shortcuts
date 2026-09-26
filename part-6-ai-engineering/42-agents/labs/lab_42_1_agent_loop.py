# %% [markdown]
# # Lab 42.1: The agent loop
#
# Task: "What are the titles of the prerequisites of lesson X?" over this course. The answer needs one lookup for the
# lesson and one per prerequisite, so tasks range from 2 to 6 steps. Tools read the real lesson files.
# The model is scripted (as in 40.3): a ReAct-style policy that thinks, acts and observes, and with probability eps per
# step makes one of the mistakes models make: answering from memory instead of looking, or repeating an action.
# 1. A traced run.
# 2. Reliability compounds: success against the number of steps, and what checks buy back.
# 3. Memory: what the context costs as the loop runs, and two ways to cut it.
# 4. Stopping: budgets and loop detection.
# 5. The same task as a plain workflow.

# %%
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from fakellm import FakeLLM, VirtualClock, count_tokens  # noqa: E402
from retrieval import lessons  # noqa: E402

L = lessons()
BY_ID = {les.id: les for les in L}


def meta_line(les):
    return next((ln for ln in les.path.read_text(encoding="utf-8").splitlines() if ln.startswith("*About")), "")


def prereqs(lid):
    line = meta_line(BY_ID[lid]).split("Lab:")[0]
    return [p for p in dict.fromkeys(re.findall(r"\b(\d{2}\.\d)\b", line)) if p in BY_ID and p != lid]


# tools, in two designs: a verbose one that returns the start of the lesson file, and a concise one
def read_lesson_verbose(lid):
    if lid not in BY_ID:
        return {"error": f"no lesson {lid}"}
    return {"text": BY_ID[lid].path.read_text(encoding="utf-8")[:1500]}


def read_lesson_concise(lid):
    if lid not in BY_ID:
        return {"error": f"no lesson {lid}"}
    return {"id": lid, "title": BY_ID[lid].title, "prerequisites": prereqs(lid)}


tasks = [(les.id, [BY_ID[p].title for p in prereqs(les.id)]) for les in L if 1 <= len(prereqs(les.id)) <= 4]
by_k = {k: [t for t in tasks if len(t[1]) == k] for k in range(1, 5)}
print(f"{len(tasks)} tasks; by number of prerequisites: {{{', '.join(f'{k}: {len(v)}' for k, v in by_k.items())}}}")


def make_brain(eps):
    all_titles = [les.title for les in L]

    def brain(messages, tools, rng):
        lid = re.search(r"lesson (\d{2}\.\d)", messages[0]["content"]).group(1)
        obs = [json.loads(m["content"][0]["content"]) for m in messages if m["role"] == "user" and isinstance(m["content"], list)]
        acts = [json.loads(m["content"]) for m in messages if m["role"] == "assistant" and m["content"].startswith("{")]
        stuck = len(acts) >= 2 and acts[-1]["input"] == acts[-2]["input"] and "already called" not in obs[-1].get("error", "")
        if stuck and rng.random() < 0.9:                                             # a loop, once started, tends to persist
            return {"tool_use": dict(acts[-1], id=f"a{len(acts)}")}
        if rng.random() < eps and acts:
            if rng.random() < 0.5:                                                   # repeats itself
                return {"tool_use": dict(acts[-1], id=f"a{len(acts)}")}
            return {"text": "ANSWER: " + " | ".join(rng.sample(all_titles, 2))}     # answers from "memory"
        if not obs:
            return {"tool_use": {"id": "a0", "name": "read_lesson", "input": {"lesson_id": lid}}}
        first = obs[0]
        todo = first.get("prerequisites") or [p for p in dict.fromkeys(re.findall(r"\b(\d{2}\.\d)\b",
                                               first.get("text", "").split("Lab:")[0])) if p != lid]
        done = {a["input"]["lesson_id"] for a in acts[1:]}
        for p in todo:
            if p not in done:
                return {"tool_use": {"id": f"a{len(acts)}", "name": "read_lesson", "input": {"lesson_id": p}}}
        titles = [o.get("title") or re.sub(r"^#\s*[0-9.]+\s*", "", o.get("text", "").splitlines()[0]) for o in obs[1:] if "error" not in o]
        return {"text": "ANSWER: " + " | ".join(dict.fromkeys(titles))}
    return brain


def run(task, eps=0.0, tool=read_lesson_concise, max_steps=12, loop_guard=False, check=False, seed=0, trace=False):
    lid, gold = task
    llm = FakeLLM(brain=make_brain(eps), clock=VirtualClock(), model="small", seed=seed * 100_003 + int(lid.replace(".", "")))
    messages = [{"role": "user", "content": f"What are the titles of the prerequisites of lesson {lid}?"}]
    seen, tokens_in, observed_titles = set(), 0, set()
    for step in range(max_steps):
        r = llm.create(messages, max_tokens=300)
        tokens_in += r["usage"]["input_tokens"]
        if r["stop_reason"] != "tool_use":
            answer = [t.strip() for t in r["content"].removeprefix("ANSWER:").split("|")]
            if check and set(answer) - observed_titles and step < max_steps - 1:
                # a verifier: every title in the answer must have come from a tool result in this conversation
                messages.append({"role": "user", "content": "Some titles in your answer weren't in any tool result. Look them up."})
                continue
            if trace:
                print(f"  [{step}] answer: {r['content'][:150]}")
            return set(answer) == set(gold), step + 1, tokens_in, "answered"
        call = r["tool_use"]
        key = json.dumps(call["input"], sort_keys=True)
        if loop_guard and key in seen:
            obs = {"error": "you already called this with the same arguments; use the earlier result"}
        else:
            obs = tool(call["input"]["lesson_id"])
        seen.add(key)
        if "error" not in obs and call["input"]["lesson_id"] != lid:
            observed_titles.add(BY_ID[call["input"]["lesson_id"]].title)
        if trace:
            print(f"  [{step}] act: read_lesson({call['input']['lesson_id']}) -> {json.dumps(obs)[:110]}")
        messages += [{"role": "assistant", "content": json.dumps(call)},
                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"], "content": json.dumps(obs)}]}]
    return False, max_steps, tokens_in, "budget exhausted"


# %% [markdown]
# ## 1. A traced run

# %%
task = by_k[3][0]
print(f"\nlesson {task[0]} ({BY_ID[task[0]].title}); gold: {task[1]}")
ok, steps, tok, status = run(task, trace=True)
print(f"correct: {ok}, {steps} model calls, {tok:,} input tokens")
assert ok

# %% [markdown]
# ## 2. Reliability compounds

# %%
print("\nsuccess rate by number of prerequisites (steps = prerequisites + 2), 3 seeds per task:")
print(f"  {'eps':>5s}  {'guards':22s}" + "".join(f"{'k=' + str(k):>8s}" for k in by_k) + "   (1-eps)^steps for k=4")
rel = {}
for eps in (0.0, 0.05, 0.15):
    for guards in (False, True):
        row = []
        for k, ts in by_k.items():
            row.append(np.mean([run(t, eps=eps, loop_guard=guards, check=guards, seed=s)[0] for t in ts for s in range(3)]))
        rel[(eps, guards)] = row
        print(f"  {eps:5.2f}  {'loop guard + verifier' if guards else 'none':22s}" + "".join(f"{v:8.2f}" for v in row)
              + f"   {(1 - eps) ** 6:.2f}")
print("every step is a chance to go wrong, so success falls with the length of the task (less steeply than")
print("(1-eps)^steps here, because some mistakes are harmless). Checks that catch a mistake and send the model back turn")
print("fatal errors into retries: a verifier that rejects titles no tool returned, a guard that refuses repeated calls.")
print("This verifier is perfect only because these mistakes are easy to detect; real ones catch some, not all.")
assert rel[(0.15, False)][-1] < rel[(0.15, False)][0] and rel[(0.15, True)][-1] > rel[(0.15, False)][-1]

# %% [markdown]
# ## 3. Memory: what the context costs

# %%
sample = by_k[4][:15]
cost = {}
for name, tool in (("verbose tool (start of the lesson file)", read_lesson_verbose), ("concise tool (title + prerequisites)", read_lesson_concise)):
    res = [run(t, tool=tool) for t in sample]
    cost[name] = (np.mean([r[0] for r in res]), np.mean([r[2] for r in res]))
    print(f"\n{name:42s} correct {cost[name][0]:.2f}, input tokens per task {cost[name][1]:8,.0f}")
v, c = cost["verbose tool (start of the lesson file)"][1], cost["concise tool (title + prerequisites)"][1]
print(f"every observation stays in the context and is re-sent at every later step, so input tokens grow with the square")
print(f"of the number of steps. Returning what the agent needs instead of everything cut the cost {v / c:.0f}x. Other")
print("levers: drop or summarize old observations (keep a short notes list), and prompt caching of the stable prefix.")
assert v > 3 * c

# %% [markdown]
# ## 4. Stopping

# %%
res = [run(t, eps=0.3, max_steps=8, seed=s) for t in by_k[2] for s in range(3)]
res_g = [run(t, eps=0.3, max_steps=8, loop_guard=True, seed=s) for t in by_k[2] for s in range(3)]
for name, rr in (("no loop guard", res), ("loop guard", res_g)):
    print(f"\neps 0.3, step budget 8, {name:14s}: correct {np.mean([r[0] for r in rr]):.2f}, budget exhausted "
          f"{np.mean([r[3] == 'budget exhausted' for r in rr]):.2f}, mean model calls {np.mean([r[1] for r in rr]):.1f}")
print("an agent needs stopping conditions it can't talk its way past: a step, token and time budget, and detection of")
print("repeated actions. What happens at the limit (hand off, return a partial answer) is a product decision.")

# %% [markdown]
# ## 5. The same task, as a workflow

# %%
def workflow(lid):
    return [BY_ID[p].title for p in prereqs(lid)]                                   # two lines of code, zero model calls


acc_wf = np.mean([set(workflow(t[0])) == set(t[1]) for t in tasks])
print(f"\nworkflow: correct {acc_wf:.2f} on all {len(tasks)} tasks (by construction: the answer is fully determined by the")
print("files, and code reads files exactly), 0 model calls, 0 tokens")
print("when the steps can be written down, write them down. Agents are for tasks whose steps depend on what you find;")
print("even then, give the fixed parts to code and the judgment to the model (42.3).")
assert acc_wf == 1.0

print("\nAll checks passed.")
