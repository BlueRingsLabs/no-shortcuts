# %% [markdown]
# # Lab 07.4: A flaky service, a careful client, and a process that shuts down properly
#
# Everything runs locally, no internet needed.
#
# 1. A small HTTP service (stdlib only) that validates input, limits body size, never leaks stack traces, and is
#    deliberately flaky: random latency with a heavy tail and occasional 503s.
# 2. A client with timeouts, retries only on retryable errors, exponential backoff with jitter, and idempotency keys.
# 3. Latency percentiles, and why p99 matters for fan-out.
# 4. A child process that handles SIGTERM gracefully (checkpoints, exits 0) vs one that gets SIGKILLed.

# %%
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# %% [markdown]
# ## 1. The service

# %%
MAX_BODY = 10_000
N_FEATURES = 3
_rng = random.Random(0)
_rng_lock = threading.Lock()
_seen_keys: dict[str, dict] = {}          # idempotency cache: key -> response
_side_effects = {"orders": 0}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):          # silence default logging to stderr
        pass

    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            return self._send(200, {"status": "ok"})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_BODY:
                return self._send(413, {"error": "body too large"})
            with _rng_lock:
                u, delay = _rng.random(), _rng.lognormvariate(-4.0, 0.9)
            time.sleep(min(delay, 0.5))                       # heavy-tailed latency
            if self.path == "/predict":
                if u < 0.15:
                    return self._send(503, {"error": "overloaded, try again"})
                data = json.loads(self.rfile.read(length))
                feats = data.get("features")
                if (not isinstance(feats, list) or len(feats) != N_FEATURES
                        or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in feats)):
                    return self._send(422, {"error": f"'features' must be {N_FEATURES} finite numbers"})
                return self._send(200, {"prediction": float(np.dot(feats, [0.5, -1.0, 2.0]))})
            if self.path == "/orders":
                key = self.headers.get("Idempotency-Key")
                if key and key in _seen_keys:
                    return self._send(200, _seen_keys[key])      # duplicate: same answer, no new side effect
                _side_effects["orders"] += 1
                resp = {"order_id": _side_effects["orders"]}
                if key:
                    _seen_keys[key] = resp
                if u < 0.3:
                    time.sleep(0.6)                              # the order WAS created, but the reply is late
                return self._send(201, resp)
            return self._send(404, {"error": "not found"})
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})
        except Exception:                                        # log internally, never leak a traceback
            return self._send(500, {"error": "internal error"})


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        # A client that timed out and closed the socket is normal life for a server, not an error worth a traceback.
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


server = QuietServer(("127.0.0.1", 0), Handler)                  # localhost only, random free port
PORT = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{PORT}"
print("service listening on", BASE)

# %% [markdown]
# ## 2. The client

# %%
class HTTPError(Exception):
    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status


def call(method, path, payload=None, timeout=0.3, headers=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise HTTPError(e.code, e.read().decode()) from None


RETRYABLE = {429, 500, 502, 503, 504}


def call_with_retries(method, path, payload=None, attempts=6, base=0.02, rng=random.Random(1), **kw):
    for i in range(attempts):
        try:
            return call(method, path, payload, **kw), i + 1
        except HTTPError as e:
            if e.status not in RETRYABLE or i == attempts - 1:
                raise
        except (TimeoutError, urllib.error.URLError):
            if i == attempts - 1:
                raise
        time.sleep(base * 2**i * rng.uniform(0.5, 1.5))           # exponential backoff with jitter


assert call("GET", "/healthz") == {"status": "ok"}
try:
    call("POST", "/predict", {"features": [1, 2]}, timeout=2)
except HTTPError as e:
    assert e.status in (422, 503)
for bad in ({"features": [1, "x", 3]}, {"features": [1, float("nan"), 3]}):
    try:
        call_with_retries("POST", "/predict", bad, timeout=2)
        raise AssertionError("invalid input must be rejected")
    except HTTPError as e:
        assert e.status == 422, "422 is not retried: retrying a bad request is pointless"
print("validation: bad inputs rejected with 422, never retried")

ok, tries = 0, []
for _ in range(60):
    resp, n_tries = call_with_retries("POST", "/predict", {"features": [1.0, 2.0, 3.0]}, timeout=2)
    assert resp["prediction"] == 4.5
    ok += 1
    tries.append(n_tries)
print(f"60/60 predictions succeeded despite ~15% 503s; mean attempts {np.mean(tries):.2f}")
assert ok == 60 and max(tries) > 1

# %% [markdown]
# ### Idempotency keys
#
# /orders sometimes replies after the client's timeout, although the order was created. Retrying without a key creates
# duplicates; retrying with the same key doesn't.

# %%
_side_effects["orders"] = 0
for _ in range(20):
    try:
        call_with_retries("POST", "/orders", {"item": "gpu"}, timeout=0.4)
    except Exception:
        pass
dupes_without_key = _side_effects["orders"]

_side_effects["orders"] = 0
for i in range(20):
    key = f"order-{i}"                                            # one key per logical order
    call_with_retries("POST", "/orders", {"item": "gpu"}, timeout=0.4, headers={"Idempotency-Key": key})
print(f"20 logical orders -> {dupes_without_key} created without keys, {_side_effects['orders']} with keys")
assert dupes_without_key > 20 and _side_effects["orders"] == 20

# %% [markdown]
# ## 3. Latency percentiles

# %%
lat = []
for _ in range(200):
    t0 = time.perf_counter()
    try:
        call("POST", "/predict", {"features": [0, 0, 0]}, timeout=2)
    except HTTPError:
        pass
    lat.append((time.perf_counter() - t0) * 1000)
p50, p95, p99 = np.percentile(lat, [50, 95, 99])
print(f"latency ms: mean {np.mean(lat):.1f}  p50 {p50:.1f}  p95 {p95:.1f}  p99 {p99:.1f}")
assert p99 > 2 * p50, "heavy tail"
fanout = 1 - 0.99**20
print(f"a page that fans out to 20 calls hits at least one p99-slow call {fanout:.0%} of the time")

# %% [markdown]
# ## 4. Graceful shutdown with SIGTERM

# %%
CHILD = textwrap.dedent("""
    import signal, sys, time, os, json
    ckpt = sys.argv[1]
    stop = False
    def on_term(signum, frame):
        global stop
        stop = True
    signal.signal(signal.SIGTERM, on_term)
    print("ready", flush=True)
    step = 0
    while True:
        time.sleep(0.01)          # "training"
        step += 1
        if stop:
            tmp = ckpt + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"step": step}, f)
            os.replace(tmp, ckpt)  # atomic rename: never a half-written checkpoint
            sys.exit(0)
""")

with tempfile.TemporaryDirectory() as tmp:
    script = os.path.join(tmp, "train.py")
    ckpt = os.path.join(tmp, "ckpt.json")
    with open(script, "w") as f:
        f.write(CHILD)

    p = subprocess.Popen([sys.executable, script, ckpt], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ready"
    time.sleep(0.3)
    p.send_signal(signal.SIGTERM)
    rc = p.wait(timeout=10)
    saved = json.load(open(ckpt))
    print(f"SIGTERM: exit code {rc}, checkpoint saved at step {saved['step']}")
    assert rc == 0 and saved["step"] > 5

    os.remove(ckpt)
    p = subprocess.Popen([sys.executable, script, ckpt], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ready"
    time.sleep(0.3)
    p.kill()                                                      # SIGKILL: no handler runs
    rc = p.wait(timeout=10)
    print(f"SIGKILL: exit code {rc}, checkpoint exists: {os.path.exists(ckpt)}")
    assert rc != 0 and not os.path.exists(ckpt)

# Configuration from the environment, validated at startup
env = {**os.environ, "MODEL_THRESHOLD": "not-a-number"}
r = subprocess.run([sys.executable, "-c", "import os; float(os.environ['MODEL_THRESHOLD'])"], env=env,
                   capture_output=True, text=True)
assert r.returncode != 0, "bad config should fail fast at startup, with a non-zero exit code"

server.shutdown()
print("\nAll checks passed.")
