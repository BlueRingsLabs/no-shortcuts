# %% [markdown]
# # Lab 25.2: A training loop you can trust
#
# 1. The Python-list bug: layers that never train.
# 2. train() vs eval(): what dropout does to evaluation.
# 3. Averaging batch averages vs counting.
# 4. A padding collate_fn for variable-length sequences, and a tiny model that uses the lengths.
# 5. DataLoader workers and a NumPy generator: identical "random" augmentations, and the fix.
# 6. The full loop with checkpointing: stop halfway, resume, and get bit-identical final weights.

# %%
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, get_worker_info
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(252)
digits = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(digits.data / 16, digits.target, test_size=0.3, random_state=0, stratify=digits.target)
X_tr, X_te = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(X_te, dtype=torch.float32)
y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)

# %% [markdown]
# ## 1. A Python list of layers

# %%
class ListMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(64, 64), nn.Linear(64, 10)]    # plain list: not registered

    def forward(self, x):
        return self.layers[1](torch.relu(self.layers[0](x)))


class GoodMLP(ListMLP):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(64, 64), nn.Linear(64, 10)])


def quick_train(model, steps=300):
    params = list(model.parameters())
    if not params:
        return None
    opt = torch.optim.SGD(params, lr=0.1)
    for _ in range(steps):
        idx = torch.randint(0, len(X_tr), (64,))
        opt.zero_grad(); nn.functional.cross_entropy(model(X_tr[idx]), y_tr[idx]).backward(); opt.step()
    return (model(X_te).argmax(1) == y_te).float().mean().item()


bad, good = ListMLP(), GoodMLP()
print(f"parameters seen by the optimizer: list version {sum(p.numel() for p in bad.parameters())}, ModuleList version {sum(p.numel() for p in good.parameters())}")
print(f"state_dict keys: list version {list(bad.state_dict())}; ModuleList {list(good.state_dict())[:2]}...")
print(f"test accuracy after training: ModuleList {quick_train(good):.3f}; the list version can't even build an optimizer (empty parameter list)")
assert sum(p.numel() for p in bad.parameters()) == 0

# %% [markdown]
# ## 2. train() vs eval()

# %%
class DropMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


m = DropMLP(); quick_train(m, 500)
with torch.no_grad():
    m.eval(); acc_eval = [(m(X_te).argmax(1) == y_te).float().mean().item() for _ in range(5)]
    m.train(); acc_train_mode = [(m(X_te).argmax(1) == y_te).float().mean().item() for _ in range(5)]
print(f"evaluating in eval() mode: {np.round(acc_eval, 3)} (deterministic)")
print(f"'evaluating' in train() mode: {np.round(acc_train_mode, 3)} (noisy and worse: dropout is on)")
assert len(set(acc_eval)) == 1 and np.mean(acc_train_mode) < acc_eval[0]

# %% [markdown]
# ## 3. Metrics over examples, not batches

# %%
correct = torch.ones(1000, dtype=torch.bool); correct[-40:] = False     # perfect except the last 40
batches = correct.split(64)
mean_of_means = np.mean([b.float().mean().item() for b in batches])
print(f"batch size 64, last batch of {len(batches[-1])}: mean of batch accuracies {mean_of_means:.4f}; true accuracy {correct.float().mean():.4f}")
assert abs(mean_of_means - 0.9375) < 1e-9

# %% [markdown]
# ## 4. Variable-length sequences

# %%
class Seqs(Dataset):                                            # label = 1 if the sequence's mean is positive
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.items = []
        for _ in range(n):
            L = int(torch.randint(3, 30, (1,), generator=g))
            s = torch.randn(L, generator=g) + (0.5 if torch.rand(1, generator=g) < 0.5 else -0.5)
            self.items.append((s, int(s.mean() > 0)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def pad_collate(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    return padded, lengths, torch.tensor(labels)


class MeanPool(nn.Module):                                      # uses lengths so padding doesn't dilute the mean
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 2)

    def forward(self, x, lengths, use_lengths=True):
        denom = lengths[:, None].float() if use_lengths else torch.full_like(lengths[:, None], x.shape[1]).float()
        return self.lin(x.sum(1, keepdim=True) / denom)


tr_loader = DataLoader(Seqs(2000, 0), batch_size=64, shuffle=True, collate_fn=pad_collate, generator=torch.Generator().manual_seed(0))
te_loader = DataLoader(Seqs(1000, 1), batch_size=256, collate_fn=pad_collate)
xb, lb, yb = next(iter(tr_loader))
print(f"a padded batch: {tuple(xb.shape)}, lengths from {lb.min().item()} to {lb.max().item()}")
accs = {}
for use_len in (True, False):
    torch.manual_seed(0)
    mp = MeanPool(); opt = torch.optim.Adam(mp.parameters(), lr=0.05)
    for epoch in range(5):
        for x_, l_, y_ in tr_loader:
            opt.zero_grad(); nn.functional.cross_entropy(mp(x_, l_, use_len), y_).backward(); opt.step()
    with torch.no_grad():
        accs[use_len] = np.mean([(mp(x_, l_, use_len).argmax(1) == y_).float().mean().item() for x_, l_, y_ in te_loader])
print(f"mean pooling with true lengths {accs[True]:.3f}; dividing by the padded length {accs[False]:.3f}")
assert accs[True] > accs[False]

# %% [markdown]
# ## 5. Workers and a NumPy generator

# %%
class Augmented(Dataset):
    def __init__(self):
        self.rng = np.random.default_rng(0)                     # created once, in the parent process

    def __len__(self):
        return 8

    def __getitem__(self, i):
        return torch.tensor(self.rng.normal())                  # "random augmentation"


def reseed(worker_id):
    info = get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2**32)   # each worker gets its own stream


# (worker processes are forked on Linux; on macOS/Windows they're spawned and this code would need to live in a module)
same = [b.tolist() for b in DataLoader(Augmented(), batch_size=2, num_workers=2)]
fixed = [b.tolist() for b in DataLoader(Augmented(), batch_size=2, num_workers=2, worker_init_fn=reseed)]
print("two workers, generator from __init__:", [[round(v, 3) for v in b] for b in same])
print("with worker_init_fn reseeding:        ", [[round(v, 3) for v in b] for b in fixed])
assert same[0] == same[1] and fixed[0] != fixed[1]

# %% [markdown]
# ## 6. Checkpoint, crash, resume: bit-identical

# %%
def set_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def make_everything(seed=0):
    set_seeds(seed)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 10))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6 * 20)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True, generator=gen)
    return model, opt, sched, loader, gen


def save_checkpoint(path, **state):
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)                                       # atomic: never a half-written checkpoint


def run(epochs, start_epoch=0, ckpt_in=None, ckpt_out=None, stop_after=None):
    model, opt, sched, loader, gen = make_everything()
    if ckpt_in:
        ck = torch.load(ckpt_in, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        gen.set_state(ck["loader_gen"]); torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["np_rng"]); random.setstate(ck["py_rng"])
        start_epoch = ck["epoch"] + 1
    for epoch in range(start_epoch, epochs):
        model.train()
        for xb_, yb_ in loader:
            opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(xb_), yb_).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        if ckpt_out:
            save_checkpoint(ckpt_out, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=epoch,
                            loader_gen=gen.get_state(), torch_rng=torch.get_rng_state(), np_rng=np.random.get_state(),
                            py_rng=random.getstate())
        if stop_after is not None and epoch == stop_after:
            return model                                        # "the machine died here"
    return model


with tempfile.TemporaryDirectory() as tmp:
    ck = Path(tmp) / "last.pt"
    uninterrupted = run(6)
    run(6, ckpt_out=ck, stop_after=2)                           # crash after epoch 3 of 6
    resumed = run(6, ckpt_in=ck)
    identical = all(torch.equal(a, b) for a, b in zip(uninterrupted.state_dict().values(), resumed.state_dict().values()))
print(f"resumed run bit-identical to the uninterrupted one: {identical}")
with torch.no_grad():
    uninterrupted.eval()
    print(f"final test accuracy {(uninterrupted(X_te).argmax(1) == y_te).float().mean():.3f}")
assert identical

print("\nAll checks passed.")
