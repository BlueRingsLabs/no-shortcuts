# %% [markdown]
# # Lab 48.1: Experiment tracking and a model registry, from scratch
#
# MLflow's core ideas in ~100 lines of SQLite, so nothing is magic when you use the real thing:
# 1. A tracker: runs with parameters, metrics, tags (code version, data hash, environment) and artifacts by hash.
# 2. A sweep, logged; querying for the best run, and what "best" hides.
# 3. Seed variance: how different is the same configuration run twice?
# 4. A registry: versions, aliases (champion, challenger), and lineage back to the run that produced each version.

# %%
import hashlib
import json
import platform
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(tempfile.mkdtemp(prefix="tracking_"))


class Tracker:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, experiment TEXT, started REAL, status TEXT);
            CREATE TABLE IF NOT EXISTS params (run INTEGER, key TEXT, value TEXT);
            CREATE TABLE IF NOT EXISTS metrics (run INTEGER, key TEXT, value REAL, step INTEGER);
            CREATE TABLE IF NOT EXISTS tags (run INTEGER, key TEXT, value TEXT);
            CREATE TABLE IF NOT EXISTS artifacts (run INTEGER, name TEXT, sha256 TEXT, path TEXT);
            CREATE TABLE IF NOT EXISTS models (name TEXT, version INTEGER, run INTEGER, artifact TEXT, created REAL);
            CREATE TABLE IF NOT EXISTS aliases (name TEXT, alias TEXT, version INTEGER, PRIMARY KEY (name, alias));
        """)

    def start(self, experiment, params, tags):
        cur = self.db.execute("INSERT INTO runs (experiment, started, status) VALUES (?, ?, 'running')", (experiment, time.time()))
        run = cur.lastrowid
        self.db.executemany("INSERT INTO params VALUES (?, ?, ?)", [(run, k, json.dumps(v)) for k, v in params.items()])
        self.db.executemany("INSERT INTO tags VALUES (?, ?, ?)", [(run, k, str(v)) for k, v in tags.items()])
        return run

    def log_metric(self, run, key, value, step=0):
        self.db.execute("INSERT INTO metrics VALUES (?, ?, ?, ?)", (run, key, float(value), step))

    def log_artifact(self, run, name, data: bytes):
        sha = hashlib.sha256(data).hexdigest()
        path = ROOT / "artifacts" / sha[:2] / sha                                   # content-addressed: same bytes, same file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.db.execute("INSERT INTO artifacts VALUES (?, ?, ?, ?)", (run, name, sha, str(path)))
        return sha

    def end(self, run, status="finished"):
        self.db.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run)); self.db.commit()

    def register(self, name, run, artifact_sha):
        v = (self.db.execute("SELECT MAX(version) FROM models WHERE name = ?", (name,)).fetchone()[0] or 0) + 1
        self.db.execute("INSERT INTO models VALUES (?, ?, ?, ?, ?)", (name, v, run, artifact_sha, time.time()))
        self.db.commit()
        return v

    def set_alias(self, name, alias, version):
        self.db.execute("INSERT OR REPLACE INTO aliases VALUES (?, ?, ?)", (name, alias, version)); self.db.commit()

    def resolve(self, name, alias):
        """What a serving system asks: which artifact is 'champion' right now, and where did it come from?"""
        return self.db.execute("""
            SELECT m.version, m.artifact, m.run, r.experiment,
                   (SELECT value FROM tags WHERE run = m.run AND key = 'data_sha256'),
                   (SELECT value FROM tags WHERE run = m.run AND key = 'git_commit')
            FROM aliases a JOIN models m ON m.name = a.name AND m.version = a.version JOIN runs r ON r.id = m.run
            WHERE a.name = ? AND a.alias = ?""", (name, alias)).fetchone()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


tracker = Tracker(ROOT / "tracking.db")

# %% [markdown]
# ## 1 and 2. A logged sweep

# %%
X, y = make_classification(n_samples=6000, n_features=30, n_informative=8, flip_y=0.08, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
data_sha = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()[:16]
ENV = {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__}


def train_run(params, seed, experiment="gbm-sweep"):
    run = tracker.start(experiment, {**params, "seed": seed}, {"git_commit": git_commit(), "data_sha256": data_sha, **ENV})
    model = HistGradientBoostingClassifier(**params, random_state=seed, early_stopping=True, validation_fraction=0.15)
    model.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    tracker.log_metric(run, "test_auc", auc)
    tracker.log_metric(run, "n_iter", model.n_iter_)
    import pickle
    sha = tracker.log_artifact(run, "model.pkl", pickle.dumps(model))            # 53.2: never load untrusted pickles
    tracker.end(run)
    return run, auc, sha


grid = [dict(learning_rate=lr, max_leaf_nodes=leaves, max_iter=300) for lr in (0.03, 0.1, 0.3) for leaves in (8, 31, 63)]
t0 = time.time()
for params in grid:
    train_run(params, seed=0)
print(f"{len(grid)} runs logged in {time.time() - t0:.1f}s to {ROOT / 'tracking.db'}")
rows = tracker.db.execute("""
    SELECT r.id, MAX(CASE WHEN p.key = 'learning_rate' THEN p.value END), MAX(CASE WHEN p.key = 'max_leaf_nodes' THEN p.value END),
           (SELECT value FROM metrics WHERE run = r.id AND key = 'test_auc') AS auc
    FROM runs r JOIN params p ON p.run = r.id WHERE r.experiment = 'gbm-sweep' GROUP BY r.id ORDER BY auc DESC LIMIT 3""").fetchall()
print("top 3 runs by test AUC:")
for rid, lr, leaves, auc in rows:
    print(f"  run {rid}: learning_rate {lr}, max_leaf_nodes {leaves}: AUC {auc:.4f}")
print("choosing the best of 9 configurations on the test set makes its score optimistic (18.1): select on a")
print("validation split, report on a test split used once. The tracker makes it easy to query; it can't make it honest.")

# %% [markdown]
# ## 3. Seed variance

# %%
best_params = dict(learning_rate=float(rows[0][1]), max_leaf_nodes=int(rows[0][2]), max_iter=300)
second_params = dict(learning_rate=float(rows[1][1]), max_leaf_nodes=int(rows[1][2]), max_iter=300)
aucs_1 = [train_run(best_params, seed=s, experiment="seeds")[1] for s in range(8)]
aucs_2 = [train_run(second_params, seed=s, experiment="seeds")[1] for s in range(8)]
gap_seed0 = rows[0][3] - rows[1][3]
print(f"\nthe top two configurations, 8 seeds each (the seed changes the early-stopping split and the binning sample):")
print(f"  best:   mean {np.mean(aucs_1):.4f}, sd {np.std(aucs_1):.4f}, range {min(aucs_1):.4f}-{max(aucs_1):.4f}")
print(f"  second: mean {np.mean(aucs_2):.4f}, sd {np.std(aucs_2):.4f}, range {min(aucs_2):.4f}-{max(aucs_2):.4f}")
print(f"gap between them with seed 0: {gap_seed0:.4f}; between their means over 8 seeds: {np.mean(aucs_1) - np.mean(aucs_2):+.4f}")
print("a leaderboard of single runs ranks noise as often as it ranks configurations. Log the seed, run several, and")
print("compare distributions; a difference smaller than the seed-to-seed spread isn't a finding.")
assert abs(np.mean(aucs_1) - np.mean(aucs_2)) < 2 * max(np.std(aucs_1), np.std(aucs_2))

# %% [markdown]
# ## 4. The registry

# %%
champ_run, champ_auc, champ_sha = train_run(best_params, seed=0, experiment="release")
v1 = tracker.register("fraud-scorer", champ_run, champ_sha)
tracker.set_alias("fraud-scorer", "champion", v1)
chall_run, chall_auc, chall_sha = train_run(second_params, seed=0, experiment="release")
v2 = tracker.register("fraud-scorer", chall_run, chall_sha)
tracker.set_alias("fraud-scorer", "challenger", v2)
for alias in ("champion", "challenger"):
    version, sha, run, exp, dsha, commit = tracker.resolve("fraud-scorer", alias)
    print(f"\n{alias:10s} -> version {version}, artifact {sha[:12]}..., from run {run} ({exp}), data {dsha}, code {commit}")
# promotion is a pointer move; rollback is moving it back. Nothing is rebuilt, nothing is lost.
tracker.set_alias("fraud-scorer", "champion", v2)
tracker.set_alias("fraud-scorer", "champion", v1)
print(f"\npromote v{v2} to champion, then roll back: champion is version {tracker.resolve('fraud-scorer', 'champion')[0]} again")
same = tracker.db.execute("SELECT COUNT(DISTINCT sha256), COUNT(*) FROM artifacts").fetchone()
print(f"{same[1]} artifacts logged, {same[0]} distinct: identical models (same config, same seed) are stored once")
print("every version points to a run, every run to its data hash, code commit and environment: that's lineage. Serving")
print("asks for 'fraud-scorer@champion', never for a file path.")
assert tracker.resolve("fraud-scorer", "champion")[0] == v1

print("\nAll checks passed.")
