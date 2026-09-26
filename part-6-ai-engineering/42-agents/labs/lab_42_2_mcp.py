# %% [markdown]
# # Lab 42.2: MCP from scratch
#
# The Model Context Protocol is JSON-RPC 2.0 between an AI application (the host, with one client per server) and
# servers that expose tools, resources and prompts. This lab speaks it by hand, against course_mcp_server.py (this
# folder), which serves this course as two tools. Protocol revision 2026-07-28: stateless, every request carries its
# protocol version and the client's capabilities in params._meta.
# 1. The wire: discovery, listing, calling, and version negotiation.
# 2. From MCP tools to model tools, and an agent loop that uses them (the simulated API from 40.1).
# 3. Several servers: name collisions and namespacing.
# 4. Trusting servers: tool definitions are prompts. Pin them, and notice when they change.
# 5. What tool definitions cost in context.

# %%
import hashlib
import json
import re
import subprocess
import sys
import threading
import queue
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "_shared"))
from fakellm import FakeLLM, VirtualClock, count_tokens  # noqa: E402

META = "io.modelcontextprotocol/"


class MCPClient:
    """One client per server: launches it, sends newline-delimited JSON-RPC on stdin, reads responses from stdout."""

    def __init__(self, name, args=(), version="2026-07-28", verbose=False):
        self.name, self.version, self.verbose, self.next_id = name, version, verbose, 1
        self.proc = subprocess.Popen([sys.executable, str(HERE / "course_mcp_server.py"), *args], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.lines = queue.Queue()
        threading.Thread(target=lambda: [self.lines.put(line) for line in self.proc.stdout], daemon=True).start()

    def request(self, method, params=None, timeout=30):
        params = dict(params or {})
        params["_meta"] = {META + "protocolVersion": self.version,
                           META + "clientInfo": {"name": "lab-42-2", "version": "0.1"},
                           META + "clientCapabilities": {}}
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params}
        self.next_id += 1
        self.proc.stdin.write(json.dumps(msg) + "\n"); self.proc.stdin.flush()
        if self.verbose:
            print("  ->", json.dumps(msg)[:150])
        try:
            resp = json.loads(self.lines.get(timeout=timeout))                   # never wait forever on a server
        except queue.Empty:
            raise TimeoutError(f"{self.name}: no response to {method} in {timeout}s")
        if self.verbose:
            print("  <-", json.dumps(resp)[:150])
        if resp.get("id") != msg["id"]:
            raise RuntimeError("response id mismatch")
        if "error" in resp:
            err = resp["error"]
            if err["code"] == -32022 and err.get("data", {}).get("supported"):
                raise VersionError(err["data"]["supported"])
            raise RuntimeError(f"{self.name}: {err['message']}")
        return resp["result"]

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


class VersionError(Exception):
    def __init__(self, supported):
        super().__init__(f"server supports {supported}")
        self.supported = supported


# %% [markdown]
# ## 1. The wire

# %%
print("an old client, speaking 2025-06-18:")
client = MCPClient("course", version="2025-06-18", verbose=True)
try:
    client.request("tools/list")
except VersionError as e:
    print(f"  version rejected; server supports {e.supported}. Retrying with {e.supported[0]}.")
    client.version = e.supported[0]
print("\ndiscovery and listing:")
disc = client.request("server/discover")
tools = client.request("tools/list")["tools"]
client.verbose = False
print(f"server {disc['_meta'][META + 'serverInfo']}, capabilities {disc['capabilities']}")
print("tools:", [(t["name"], list(t["inputSchema"]["properties"])) for t in tools])
res = client.request("tools/call", {"name": "course_search", "arguments": {"query": "KV cache decoding", "k": 2}})
print("\ntools/call course_search ->", res["content"][0]["text"][:200].replace("\n", " | "), "...")
bad = client.request("tools/call", {"name": "course_search", "arguments": {"query": "x", "k": 50}})
print("a bad argument comes back as a tool result with isError, which the model can read and fix:", bad["content"][0]["text"])
assert client.version == "2026-07-28" and len(tools) == 2 and bad.get("isError")

# %% [markdown]
# ## 2. MCP tools as model tools, in an agent loop
#
# The host turns each MCP tool into a tool definition for the model API (inputSchema -> input_schema), and routes
# the model's tool calls back to the right server. The "model" here is a scripted brain, as in 40.3: it searches,
# reads the most relevant section, and answers from what it read.

# %%
def to_model_tool(server, t):
    return {"name": f"{server}__{t['name']}", "description": t["description"], "input_schema": t["inputSchema"]}


def brain(messages, model_tools, rng):
    question = messages[0]["content"]
    results = [m["content"][0]["content"] for m in messages if m["role"] == "user" and isinstance(m["content"], list)]
    if not results:
        return {"tool_use": {"id": "t0", "name": "course__course_search", "input": {"query": question, "k": 3}}}
    if len(results) == 1:
        lid, heading = re.search(r"\[(\d{2}\.\d) [^/]+/ ([^\]]+)\]", results[0]).groups()
        return {"tool_use": {"id": "t1", "name": "course__course_read_section", "input": {"lesson_id": lid, "heading": heading}}}
    sentence = next((s for s in re.split(r"(?<=[.!?])\s+", results[1]) if "stores" in s), results[1][:200])
    return {"text": "From the course: " + " ".join(sentence.split())}


servers = {"course": client}
registry = {to_model_tool(name, t)["name"]: (name, t["name"]) for name, c in servers.items() for t in tools}
model_tools = [to_model_tool("course", t) for t in tools]
llm = FakeLLM(brain=brain, clock=VirtualClock(), model="small")
messages = [{"role": "user", "content": "What does the KV cache store, and why does it speed up generation?"}]
for step in range(6):
    r = llm.create(messages, tools=model_tools, max_tokens=400)
    if r["stop_reason"] != "tool_use":
        print(f"\n[{step}] answer: {r['content'][:300]}")
        break
    call = r["tool_use"]
    server, tool = registry[call["name"]]
    out = servers[server].request("tools/call", {"name": tool, "arguments": call["input"]})
    text = out["content"][0]["text"]
    print(f"[{step}] {call['name']}({json.dumps(call['input'])}) -> {len(text)} characters")
    messages += [{"role": "assistant", "content": json.dumps(call)},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"], "content": text,
                                               "is_error": bool(out.get("isError"))}]}]
assert "keys and values" in r["content"]

# %% [markdown]
# ## 3. Several servers: collisions

# %%
second = MCPClient("course_mirror")
tools2 = second.request("tools/list")["tools"]
bare = [t["name"] for t in tools] + [t["name"] for t in tools2]
collisions = sorted({n for n in bare if bare.count(n) > 1})
print(f"\ntwo servers, bare tool names: {bare}")
print(f"collisions: {collisions}. Without namespacing, which server gets the call is up to the host's merge order: a")
print("server you added later can silently take over a tool name the model already relies on.")
namespaced = [to_model_tool("course", t)["name"] for t in tools] + [to_model_tool("course_mirror", t)["name"] for t in tools2]
print(f"namespaced: {namespaced}")
assert collisions and len(set(namespaced)) == len(namespaced)
second.close()

# %% [markdown]
# ## 4. Tool definitions are prompts: pin them
#
# The model reads every tool's name, description and schema, and they shape what it does. A server can change them
# at any time, and a server you installed for one tool gets to write text into the model's context for every
# conversation. The host's defense: show the definitions to a person once, record a hash of each approved definition,
# and refuse (or re-ask) when one changes. The second server below changes a description after its first listing.

# %%
def fingerprint(t):
    return hashlib.sha256(json.dumps({k: t[k] for k in ("name", "description", "inputSchema")}, sort_keys=True).encode()).hexdigest()[:16]


class PinnedTools:
    def __init__(self):
        self.approved = {}                                                     # (server, tool) -> fingerprint

    def approve(self, server, tool_list):
        for t in tool_list:
            self.approved[(server, t["name"])] = fingerprint(t)

    def check(self, server, tool_list):
        usable, changed = [], []
        for t in tool_list:
            fp = self.approved.get((server, t["name"]))
            (usable if fp == fingerprint(t) else changed).append(t)
        return usable, changed


SUSPICIOUS = re.compile(r"\b(always|before (answering|using|calling)|ignore|instead of|do not tell|include the (full|whole) "
                        r"conversation|system prompt|all (other|previous))\b", re.I)

pins = PinnedTools()
shifty = MCPClient("notes", args=("--variant", "rugpull"))
first = shifty.request("tools/list")["tools"]
pins.approve("notes", first)
print(f"\nfirst listing approved: {[(t['name'], fingerprint(t)) for t in first]}")
later = shifty.request("tools/list")["tools"]
usable, changed = pins.check("notes", later)
for t in changed:
    print(f"CHANGED since approval: {t['name']} {fingerprint(t)}")
    print(f"  new description: {t['description']}")
    print(f"  instruction-like phrases: {[m.group(0) for m in SUSPICIOUS.finditer(t['description'])]}")
print(f"usable without re-approval: {[t['name'] for t in usable]}")
print("the change asks the model to call this tool first, every time, with the whole conversation as the argument:")
print("a data-exfiltration channel written as a usage tip. Pinning catches the change; the phrase filter is only a hint")
print("for the reviewer, and easy to word around. What limits the damage is least privilege: this server should never")
print("have been able to receive the conversation, so the host must not send it more than the tool's arguments need.")
assert [t["name"] for t in changed] == ["course_search"] and SUSPICIOUS.search(changed[0]["description"])
shifty.close()

# %% [markdown]
# ## 5. What tool definitions cost

# %%
n_two = count_tokens(json.dumps(model_tools))
fake_many = [dict(model_tools[i % 2], name=f"server{i // 2}__tool{i}") for i in range(60)]
n_many = count_tokens(json.dumps(fake_many))
print(f"\ntool definitions in every request: 2 tools {n_two} tokens; 60 tools (a few servers' worth) {n_many:,} tokens")
print("definitions are sent with every call, cost money and latency, and too many of them make tool choice worse.")
print("Hosts connected to many servers load tool definitions progressively (a search over tools, then the definitions")
print("of the few that match), and prompt caching (40.1) makes the stable part cheaper.")
client.close()

print("\nAll checks passed.")
