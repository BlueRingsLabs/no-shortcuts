# %% [markdown]
# # Lab 53.2: The ML supply chain
#
# 1. Why loading a pickle is running code: a harmless demonstration (the "payload" appends to a list in this lab),
#    what torch.load(weights_only=True) does about it, and a scanner that inspects a pickle without loading it.
# 2. A format that can't run code: tensors as a JSON header plus raw bytes (the idea behind safetensors).
# 3. Integrity: pinning artifact hashes, and signatures that detect tampering.
# 4. Knowing what you run: a software bill of materials, and catching typo-squatted dependency names.

# %%
import hashlib
import hmac
import io
import json
import pickle
import pickletools
import secrets
import difflib
from importlib import metadata

import numpy as np
import torch

EVENTS = []


def record_event(msg):
    """Stands in for whatever a malicious file would run. Here: it only appends to a list."""
    EVENTS.append(msg)
    return {"weights": [0.1, 0.2]}


class LooksLikeAModel:
    def __reduce__(self):
        return (record_event, ("code ran during unpickling",))                    # pickle will CALL this on load


# %% [markdown]
# ## 1. Pickle runs code

# %%
blob = pickle.dumps(LooksLikeAModel())
obj = pickle.loads(blob)
print(f"after pickle.loads: EVENTS = {EVENTS}; the loaded object is {obj}")
print("unpickling calls whatever callable the file names, with whatever arguments it stores. A model file from the")
print("internet in pickle format (.pkl, .pt, .bin, .joblib) is a program; loading it runs it with your permissions.")
assert EVENTS == ["code ran during unpickling"]

buf = io.BytesIO(); torch.save({"layer.weight": torch.randn(3, 3)}, buf)
torch_blob = buf.getvalue()
buf2 = io.BytesIO(); torch.save(LooksLikeAModel(), buf2)
for name, data in (("a plain state_dict", torch_blob), ("the file with a callable", buf2.getvalue())):
    try:
        torch.load(io.BytesIO(data), weights_only=True)
        print(f"torch.load(weights_only=True) on {name}: loaded")
    except Exception as e:
        print(f"torch.load(weights_only=True) on {name}: refused ({type(e).__name__})")
print(f"EVENTS is still {len(EVENTS)} entry long: the restricted unpickler refused before calling anything.")
assert len(EVENTS) == 1


def scan_pickle(data, allowed_prefixes=("torch.", "collections.OrderedDict", "numpy.core.multiarray._reconstruct",
                                         "numpy.dtype", "numpy.ndarray", "builtins.set", "_codecs.encode")):
    """List the globals a pickle would import, without executing it; flag anything outside an allowlist."""
    if data[:2] == b"PK":                                                           # torch.save zips; the pickle is inside
        import zipfile
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            data = z.read(next(n for n in z.namelist() if n.endswith("data.pkl")))
    found, strings = [], []
    for op, arg, _ in pickletools.genops(data):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
            strings.append(arg)
        elif op.name == "GLOBAL":
            found.append(arg.replace(" ", "."))
        elif op.name == "STACK_GLOBAL":
            found.append(f"{strings[-2]}.{strings[-1]}")
    return found, [g for g in found if not g.startswith(allowed_prefixes)]


for name, data in (("the state_dict", torch_blob), ("the file with a callable", blob)):
    found, bad = scan_pickle(data)
    print(f"scan of {name}: imports {sorted(set(found))}; outside the allowlist: {bad or 'none'}")
assert scan_pickle(blob)[1] and not scan_pickle(torch_blob)[1]
print("scanners like this (picklescan, and the checks model hubs run) catch known-bad imports. They can be evaded by")
print("clever payloads; the fix is not to load pickles from untrusted sources at all.")

# %% [markdown]
# ## 2. A format that can't run code

# %%
def save_tensors(tensors: dict) -> bytes:
    header, chunks, offset = {}, [], 0
    for name, t in tensors.items():
        a = t.detach().cpu().contiguous().numpy()
        b = a.tobytes()
        header[name] = {"dtype": str(a.dtype), "shape": list(a.shape), "offsets": [offset, offset + len(b)]}
        chunks.append(b); offset += len(b)
    h = json.dumps(header).encode()
    return len(h).to_bytes(8, "little") + h + b"".join(chunks)


def load_tensors(data: bytes) -> dict:
    n = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8 + n])                                              # JSON: data only, no code
    body = data[8 + n:]
    out = {}
    for name, h in header.items():
        s, e = h["offsets"]
        out[name] = torch.from_numpy(np.frombuffer(body[s:e], dtype=h["dtype"]).reshape(h["shape"]).copy())
    return out


model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
data = save_tensors(model.state_dict())
restored = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
restored.load_state_dict(load_tensors(data))
x = torch.randn(4, 8)
print(f"\nsafe format: {len(data):,} bytes, header {json.loads(data[8:8 + int.from_bytes(data[:8], 'little')])['0.weight']}")
print(f"round trip identical: {torch.equal(model(x), restored(x))}")
print("a header that is only data, and bytes that are only numbers: loading can't execute anything, and it can be")
print("memory-mapped. That's safetensors, now the default on the Hugging Face hub; use it (or ONNX) for models you share.")
assert torch.equal(model(x), restored(x))

# %% [markdown]
# ## 3. Integrity: hashes and signatures

# %%
artifact = data
pinned = hashlib.sha256(artifact).hexdigest()
key = secrets.token_bytes(32)                                                        # stands in for a signing key
signature = hmac.new(key, artifact, hashlib.sha256).hexdigest()
tampered = bytearray(artifact); tampered[-5] ^= 0x01                                 # one bit flipped in one weight
for name, blob_ in (("original", bytes(artifact)), ("one bit changed", bytes(tampered))):
    ok_hash = hashlib.sha256(blob_).hexdigest() == pinned
    ok_sig = hmac.compare_digest(hmac.new(key, blob_, hashlib.sha256).hexdigest(), signature)
    print(f"{name:16s}: hash matches the pinned one: {ok_hash}; signature valid: {ok_sig}")
print("pin the hash of every artifact you depend on (in the registry, 48.1, or a lock file) and verify it before loading.")
print("Signatures add who: Sigstore and cosign sign models and images with short-lived keys tied to an identity, so you")
print("can require 'signed by our CI' before anything runs in production. A single flipped bit is caught; so is a swap.")

# %% [markdown]
# ## 4. Knowing what you run

# %%
dists = sorted(((d.metadata["Name"], d.version) for d in metadata.distributions() if d.metadata["Name"]), key=lambda t: t[0].lower())
sbom = {"bomFormat": "CycloneDX", "specVersion": "1.5",
        "components": [{"type": "library", "name": n, "version": v, "purl": f"pkg:pypi/{n.lower()}@{v}"} for n, v in dists]}
print(f"\nSBOM of this environment: {len(sbom['components'])} Python packages, e.g. {sbom['components'][:2]}")
print("match every component against vulnerability databases (OSV, GitHub advisories) on every build: pip-audit, Trivy.")

KNOWN = ["numpy", "torch", "scikit-learn", "pandas", "requests", "transformers", "tokenizers", "safetensors", "matplotlib"]
requested = ["numpy", "torch", "scikit-learn", "reqeusts", "transfomers", "safetensor", "pandas"]
for name in requested:
    if name not in KNOWN:
        close = difflib.get_close_matches(name, KNOWN, n=1, cutoff=0.8)
        print(f"  requirement '{name}' is not on the allowlist" + (f"; did you mean '{close[0]}'? (possible typo-squat)" if close else ""))
print("attackers publish packages one typo away from popular ones. Install from a curated internal index or an allowlist,")
print("pin versions with hashes (48.2), and review new dependencies like code, because they are code.")

print("\nAll checks passed.")
