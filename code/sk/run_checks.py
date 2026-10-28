import sys, warnings, importlib
warnings.simplefilter("ignore")
from sklearn.utils.estimator_checks import check_estimator
mod, cls = sys.argv[1], sys.argv[2]
est = getattr(importlib.import_module(mod), cls)()
res = check_estimator(est, on_fail=None)
status = {}
for r in res:
    status[r["status"]] = status.get(r["status"], 0) + 1
print(cls, status)
for r in res:
    if r["status"] == "failed":
        lines = str(r["exception"]).strip().splitlines() if r["exception"] else []; msg = lines[0][:150] if lines else ""
        print(f"  FAILED {r['check_name']}: {type(r['exception']).__name__}: {msg}")
