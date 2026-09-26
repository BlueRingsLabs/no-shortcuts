# %% [markdown]
# # Lab 49.1: Serving patterns, and where p99 comes from
#
# A real HTTP model server on localhost (standard library only; FastAPI in the lesson), a load generator, and:
# 1. Batch vs online: the cost per prediction of each.
# 2. Latency percentiles as load rises: queueing, and why p99 explodes before the CPU looks busy.
# 3. Dynamic batching: collect requests for a few milliseconds and run them together, on a model heavy enough to benefit.
# Timings depend on this machine and on whatever else runs on it; the shapes of the curves are the point.

# %%
import json
import multiprocessing as mp
import queue
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)


def mlp(hidden):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)).eval()


MODELS = {"small": mlp(512), "large": mlp(4096)}                                   # 0.3M and 17M parameters
model = MODELS["small"]


@torch.no_grad()
def predict(batch):
    return model(torch.as_tensor(np.asarray(batch, dtype=np.float32)))[:, 0].tolist()


# %% [markdown]
# ## 1. Batch vs online, per prediction

# %%
X = np.random.default_rng(0).normal(size=(20_000, 64)).astype(np.float32)
t0 = time.perf_counter(); predict(X); t_batch = (time.perf_counter() - t0) / len(X)
t0 = time.perf_counter()
for row in X[:2000]:
    predict(row[None])
t_single = (time.perf_counter() - t0) / 2000
print(f"batch scoring: {t_batch * 1e6:.1f} us per prediction; one at a time, in process: {t_single * 1e6:.1f} us "
      f"({t_single / t_batch:.0f}x more)")
print("if predictions can be computed before they're needed (nightly scores, recommendations refreshed hourly), batch")
print("inference is simpler and far cheaper: no server, no latency budget, results in a table.")

# %% [markdown]
# ## 2. An online server under load

# %%
class Batcher:
    """Dynamic batching: requests wait up to max_wait for others, then run together (as Triton, TorchServe, vLLM do)."""

    def __init__(self, max_batch=32, max_wait=0.002):
        self.q, self.max_batch, self.max_wait = queue.Queue(), max_batch, max_wait
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while True:
            items = [self.q.get()]
            deadline = time.perf_counter() + self.max_wait
            while len(items) < self.max_batch:
                try:
                    items.append(self.q.get(timeout=max(0.0, deadline - time.perf_counter())))
                except queue.Empty:
                    break
            outs = predict([x for x, _ in items])
            for (_, slot), y in zip(items, outs):
                slot.put(y)

    def __call__(self, x):
        slot = queue.Queue(maxsize=1)
        self.q.put((x, slot))
        return slot.get()


class Handler(BaseHTTPRequestHandler):
    disable_nagle_algorithm = True                                                   # else: 40 ms delayed-ACK stalls
    batcher = None
    lock = threading.Lock()                                                          # one model, one core: requests take turns

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if Handler.batcher is not None:
            y = Handler.batcher(body["features"])
        else:
            with Handler.lock:
                y = predict([body["features"]])[0]
        out = json.dumps({"score": y}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

    def log_message(self, *a):
        pass


class Server(ThreadingHTTPServer):
    request_queue_size = 256                                                          # the listen backlog; the default, 5,
    daemon_threads = True                                                             # resets connections under a burst


def serve(port_queue, batching, which):
    """Runs in its own process, like a real server: the load generator must not share its interpreter."""
    global model
    model = MODELS[which]
    Handler.batcher = Batcher() if batching else None
    srv = Server(("127.0.0.1", 0), Handler)
    port_queue.put(srv.server_address[1])
    srv.serve_forever()


def start_server(batching, which):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=serve, args=(q, batching, which), daemon=True)
    proc.start()
    return proc, f"http://127.0.0.1:{q.get(timeout=30)}/predict"


payload = json.dumps({"features": X[0].tolist()}).encode()
URL = None


def one_request():
    t0 = time.perf_counter()
    req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"})
    json.loads(urllib.request.urlopen(req, timeout=10).read())
    return time.perf_counter() - t0


def load_test(concurrency, n=400):
    with ThreadPoolExecutor(concurrency) as ex:
        t0 = time.perf_counter()
        lat = list(ex.map(lambda _: one_request(), range(n)))
        wall = time.perf_counter() - t0
    lat = np.array(lat) * 1000
    return n / wall, np.percentile(lat, 50), np.percentile(lat, 95), np.percentile(lat, 99)


def run_load(label, batching, which):
    global URL
    proc, URL = start_server(batching, which)
    for _ in range(20):
        one_request()                                                                 # warm up
    print(f"  {label}:")
    print(f"    {'clients':>7s} {'req/s':>8s} {'p50 ms':>8s} {'p95 ms':>8s} {'p99 ms':>8s}")
    out = {}
    for c in (1, 4, 16, 32):
        out[c] = load_test(c)
        rps, p50, p95, p99 = out[c]
        print(f"    {c:7d} {rps:8.0f} {p50:8.1f} {p95:8.1f} {p99:8.1f}")
    proc.terminate(); proc.join()
    return out


print("\nclosed-loop load test: C clients each send a request as soon as the previous one returns")
small = run_load("small model, one request at a time", False, "small")
print(f"the model takes {t_single * 1e6:.0f} us per request; one client gets a response in {small[1][1]:.1f} ms:")
print("connection setup, HTTP parsing, JSON and Python threads cost far more than the model. Profile before you")
print("optimize the model (33.3).")
print("once requests arrive faster than one worker can serve them, they queue, and latency is mostly waiting: with")
print("throughput capped, p50 grows with the number of clients (Little's law: in-flight = throughput x latency).")
if small[32][0] < 0.7 * small[1][0]:
    print(f"Here throughput even falls, from {small[1][0]:.0f} to {small[32][0]:.0f} req/s: dozens of Python threads contending for one")
    print("interpreter and a lock cost more than the work they do.")
print("The tail grows fastest: p99 is where queueing, garbage collection, a slow neighbor and an unlucky scheduling decision")
print("add up.")
assert small[1][1] > 3 * t_single * 1000                                             # the request, not the model, is the cost
assert small[32][1] > 4 * small[1][1]                                                # queueing: 32 clients wait far longer

# %% [markdown]
# ## 3. Dynamic batching
#
# Batching pays when the model's cost per call is large and grows slowly with the batch: on a GPU, or here, a larger
# network whose one-row forward pass is dominated by reading its weights. For the small model above it can't help much:
# the model is a small fraction of each request.

# %%
model = MODELS["large"]
with torch.no_grad():
    x1, x32 = torch.as_tensor(X[:1]), torch.as_tensor(X[:32])
    for _ in range(5):
        model(x1); model(x32)
    t0 = time.perf_counter(); [model(x1) for _ in range(50)]; t_one = (time.perf_counter() - t0) / 50
    t0 = time.perf_counter(); [model(x32) for _ in range(20)]; t_32 = (time.perf_counter() - t0) / 20
print(f"\nlarge model (17M parameters): one row {t_one * 1e3:.2f} ms, a batch of 32 {t_32 * 1e3:.2f} ms "
      f"({t_32 / 32 * 1e3:.2f} ms per row)")
plain = run_load("large model, one request at a time", False, "large")
batched = run_load("large model, dynamic batching (2 ms, up to 32)", True, "large")
print(f"at 32 clients: {plain[32][0]:.0f} req/s one at a time, {batched[32][0]:.0f} req/s with batching; "
      f"p99 {plain[32][3]:.0f} ms against {batched[32][3]:.0f} ms.")
print("the server can't run requests faster than one model call each, so without batching its throughput is capped")
print("near 1 / (time per call). Batching serves many requests per call: throughput rises, and at high load the shorter")
print("queue cuts latency too. At low load it costs up to the batching window.")
assert t_32 / 32 < t_one / 2                                                         # the premise: batching amortizes
assert batched[32][0] > 1.5 * plain[32][0]

print("\nAll checks passed.")
