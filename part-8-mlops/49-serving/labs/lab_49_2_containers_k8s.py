# %% [markdown]
# # Lab 49.2: Containers and Kubernetes for ML
#
# No cluster needed: the artifacts are text, and the dynamics can be simulated.
# 1. A Dockerfile and a Kubernetes Deployment for a model server, and a linter that checks them for the mistakes
#    that cause most incidents (unpinned images, root, no resource limits, no health probes, model baked in wrong).
# 2. Autoscaling: the Horizontal Pod Autoscaler's rule, a traffic spike, and cold starts. What happens to latency
#    while new replicas load a 2 GB model, and what minimum replicas buy.

# %%
import math
import re

import numpy as np
import yaml

# %% [markdown]
# ## 1. Build and deploy artifacts, linted

# %%
DOCKERFILE_NAIVE = """\
FROM python:latest
COPY . /app
RUN pip install -r /app/requirements.txt
CMD python /app/serve.py
"""

DOCKERFILE_GOOD = """\
# build stage: resolve and install pinned dependencies
FROM python:3.12-slim@sha256:1f5d3c2e4a8b9c0d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d AS build
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock --target /deps
# runtime stage: no compilers, no pip cache, non-root
FROM python:3.12-slim@sha256:1f5d3c2e4a8b9c0d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d
RUN useradd --uid 10001 --create-home app
COPY --from=build /deps /usr/local/lib/python3.12/site-packages
COPY --chown=app serve.py /app/
USER 10001
EXPOSE 8080
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz')"
CMD ["python", "/app/serve.py"]
"""

DEPLOYMENT_NAIVE = """\
apiVersion: apps/v1
kind: Deployment
metadata: {name: scorer}
spec:
  replicas: 1
  selector: {matchLabels: {app: scorer}}
  template:
    metadata: {labels: {app: scorer}}
    spec:
      containers:
      - name: scorer
        image: registry.example.test/scorer:latest
        env:
        - {name: API_KEY, value: "sk-live-1234"}
"""

DEPLOYMENT_GOOD = """\
apiVersion: apps/v1
kind: Deployment
metadata: {name: scorer, labels: {app: scorer}}
spec:
  replicas: 3
  selector: {matchLabels: {app: scorer}}
  strategy: {type: RollingUpdate, rollingUpdate: {maxUnavailable: 0, maxSurge: 1}}
  template:
    metadata: {labels: {app: scorer}}
    spec:
      securityContext: {runAsNonRoot: true, runAsUser: 10001}
      containers:
      - name: scorer
        image: registry.example.test/scorer@sha256:9c1b2a3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f9
        ports: [{containerPort: 8080}]
        env:
        - {name: MODEL_URI, value: "models:/fraud-scorer@champion"}
        - name: API_KEY
          valueFrom: {secretKeyRef: {name: scorer-secrets, key: api-key}}
        resources:
          requests: {cpu: "2", memory: 4Gi}
          limits: {memory: 4Gi}
        startupProbe: {httpGet: {path: /healthz, port: 8080}, periodSeconds: 5, failureThreshold: 60}
        readinessProbe: {httpGet: {path: /ready, port: 8080}, periodSeconds: 5}
        livenessProbe: {httpGet: {path: /healthz, port: 8080}, periodSeconds: 10, failureThreshold: 3}
        securityContext: {allowPrivilegeEscalation: false, readOnlyRootFilesystem: true}
"""


def lint_dockerfile(text):
    issues = []
    froms = re.findall(r"^FROM\s+(\S+)", text, re.M)
    if any(":latest" in f or (":" not in f and "@" not in f) for f in froms):
        issues.append("base image not pinned (latest or no tag)")
    if froms and not all("@sha256:" in f for f in froms):
        issues.append("base image not pinned by digest")
    if not re.search(r"^USER\s+(?!root|0\b)", text, re.M):
        issues.append("runs as root")
    if re.search(r"^COPY\s+\.\s", text, re.M):
        issues.append("COPY . copies the whole context (secrets, data, .git) into the image")
    if "requirements" in text and "--require-hashes" not in text and ".lock" not in text:
        issues.append("dependencies not locked")
    if re.search(r"^CMD\s+[^\[]", text, re.M):
        issues.append("shell-form CMD: signals don't reach the process, so shutdowns aren't graceful")
    if "HEALTHCHECK" not in text:
        issues.append("no health check")
    return issues


def lint_deployment(text):
    d = yaml.safe_load(text)
    spec = d["spec"]; pod = spec["template"]["spec"]
    issues = []
    if spec.get("replicas", 1) < 2:
        issues.append("a single replica: every deploy or node failure is an outage")
    for c in pod["containers"]:
        img = c["image"]
        if img.endswith(":latest") or "@sha256:" not in img:
            issues.append(f"{c['name']}: image not pinned by digest")
        res = c.get("resources", {})
        if "requests" not in res:
            issues.append(f"{c['name']}: no resource requests (the scheduler can't place it sensibly)")
        if "memory" not in res.get("limits", {}):
            issues.append(f"{c['name']}: no memory limit (one leak takes the node down)")
        for probe in ("readinessProbe", "livenessProbe"):
            if probe not in c:
                issues.append(f"{c['name']}: no {probe}")
        if "startupProbe" not in c and "livenessProbe" in c:
            issues.append(f"{c['name']}: liveness without a startup probe kills slow-loading models")
        for e in c.get("env", []):
            if "value" in e and re.search(r"(key|secret|token|password)", e["name"], re.I):
                issues.append(f"{c['name']}: secret {e['name']} in plain text in the manifest")
    if not pod.get("securityContext", {}).get("runAsNonRoot"):
        issues.append("pod may run as root")
    return issues


for name, text, lint in (("naive Dockerfile", DOCKERFILE_NAIVE, lint_dockerfile), ("good Dockerfile", DOCKERFILE_GOOD, lint_dockerfile),
                         ("naive Deployment", DEPLOYMENT_NAIVE, lint_deployment), ("good Deployment", DEPLOYMENT_GOOD, lint_deployment)):
    issues = lint(text)
    print(f"{name}: {len(issues)} issue(s)")
    for i in issues:
        print(f"  - {i}")
print("real linters (hadolint, kube-linter, Checkov, Trivy for vulnerabilities) check hundreds of rules. The ones above")
print("cause most ML serving incidents: an image that changed under the same tag, a liveness probe that kills a pod")
print("still loading its model, no memory limit, and a secret committed to a manifest.")
assert not lint_dockerfile(DOCKERFILE_GOOD) and not lint_deployment(DEPLOYMENT_GOOD)
assert len(lint_dockerfile(DOCKERFILE_NAIVE)) >= 5 and len(lint_deployment(DEPLOYMENT_NAIVE)) >= 5

# %% [markdown]
# ## 2. Autoscaling and cold starts
#
# The HPA rule: desired = ceil(current x observed metric / target), checked every 15 s, here on requests per second
# per replica. Each replica serves up to 50 requests/s. A new replica needs time before it's ready: pull the image,
# load the model. Traffic jumps from 150 to 600 requests/s at t = 300 s.

# %%
def simulate(min_replicas, startup_s, target_rps=35, capacity=50, horizon=900, step=1):
    ready, starting = min_replicas, []                                           # starting: times at which each becomes ready
    backlog, dropped, lat = 0.0, 0.0, []
    for t in range(0, horizon, step):
        rps = 150 if t < 300 else 600
        ready += sum(1 for s in starting if s <= t)
        starting = [s for s in starting if s > t]
        served = min(rps + backlog, ready * capacity)
        backlog = rps + backlog - served
        if backlog > 2000:                                                        # requests time out after the queue fills
            dropped += backlog - 2000; backlog = 2000
        util = (rps + backlog) / max(ready, 1)
        lat.append(0.02 + backlog / max(served, 1e-9))                            # service time + queueing (Little's law)
        if t % 15 == 0:
            desired = max(min_replicas, math.ceil(ready * util / target_rps))
            new = desired - ready - len(starting)
            starting += [t + startup_s] * max(0, new)
    lat = np.array(lat)
    spike = lat[300:]
    return ready, dropped, np.percentile(spike, 50), np.percentile(spike, 99), np.argmax(lat[300:] < 0.05)


print("\ntraffic 150 -> 600 requests/s at t = 300 s; each replica serves 50/s; HPA target 35/s per replica")
print(f"  {'setup':46s} {'replicas':>8s} {'dropped':>8s} {'p50 s':>7s} {'p99 s':>7s} {'seconds to recover':>19s}")
scen = {}
for name, mn, st in (("min 5 replicas, model in the image (20 s start)", 5, 20),
                     ("min 5, model downloaded at start (180 s)", 5, 180),
                     ("min 12 (headroom for the spike), 180 s start", 12, 180)):
    scen[name] = simulate(mn, st)
    r, d, p50, p99, rec = scen[name]
    print(f"  {name:46s} {r:8d} {d:8,.0f} {p50:7.2f} {p99:7.2f} {rec:19d}")
print("autoscaling reacts to load it has already failed to serve, then waits for new replicas to start. With a model")
print("that takes minutes to load, the spike is over, or the users gone, before help arrives. Make replicas start fast")
print("(small images, models on local disk or baked in, lazy loading), keep headroom for known peaks, scale on leading")
print("signals (queue length, scheduled events), and shed load gracefully rather than letting queues time out.")
assert scen["min 5, model downloaded at start (180 s)"][1] > scen["min 5 replicas, model in the image (20 s start)"][1]
assert scen["min 12 (headroom for the spike), 180 s start"][1] == 0

print("\nAll checks passed.")
