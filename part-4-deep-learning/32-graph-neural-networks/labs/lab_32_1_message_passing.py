# %% [markdown]
# # Lab 32.1: Message passing and graph neural networks
#
# 1. A graph with communities (a stochastic block model) and noisy node features. 20 labeled nodes per class.
# 2. Features only (MLP), graph only (label propagation), both (a GCN written from scratch).
# 3. Permutation equivariance: renumber the nodes and nothing changes but the order.
# 4. Over-smoothing: deeper GCNs, with and without residual connections; how alike the node vectors become.
# 5. Heterophily: when neighbors tend to be *different*, averaging them hurts. GraphSAGE keeps a node's own features apart.

# %%
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(321)
torch.set_num_threads(4)


def sbm(n_per=150, k=4, p_in=0.05, p_out=0.005, feat_dim=16, feat_signal=0.6, seed=0):
    """Stochastic block model: k communities, edges inside with prob p_in and across with p_out. Node features:
    a class-specific mean times feat_signal, plus unit Gaussian noise: informative, not enough on their own."""
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(k).repeat_interleave(n_per)
    n = len(y)
    P = torch.where(y[:, None] == y[None, :], p_in, p_out)
    A = (torch.rand(n, n, generator=g) < P).float().triu(1)
    A = A + A.T
    means = torch.randn(k, feat_dim, generator=g)
    X = feat_signal * means[y] + torch.randn(n, feat_dim, generator=g)
    return A, X, y


def split(y, per_class=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    train = torch.cat([torch.nonzero(y == c)[:, 0][torch.randperm(int((y == c).sum()), generator=g)[:per_class]] for c in y.unique()])
    mask = torch.zeros(len(y), dtype=torch.bool); mask[train] = True
    return mask


A, X, y = sbm()
train_mask = split(y)
test_mask = ~train_mask
deg = A.sum(1)
homophily = (A * (y[:, None] == y[None, :])).sum() / A.sum()
print(f"{len(y)} nodes, {int(A.sum() / 2)} edges, mean degree {deg.mean():.1f}, isolated nodes {int((deg == 0).sum())}")
print(f"edge homophily (fraction of edges inside a class): {homophily:.2f}; labeled nodes: {int(train_mask.sum())}")

# %% [markdown]
# ## 2. Features, graph, both

# %%
def norm_adj(A):
    """The GCN propagation matrix: D^-1/2 (A + I) D^-1/2 (Kipf & Welling's renormalization trick)."""
    A_hat = A + torch.eye(len(A))
    d = A_hat.sum(1)
    return (A_hat / d.sqrt()[:, None] / d.sqrt()[None, :]).to_sparse()


class GCN(nn.Module):
    def __init__(self, i, h, o, layers=2, residual=False, dropout=0.5):
        super().__init__()
        dims = [i] + [h] * (layers - 1) + [o]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.residual, self.dropout = residual, dropout

    def forward(self, X, S, return_hidden=False):
        h = X
        for li, lin in enumerate(self.lins):
            h_in = h
            h = torch.sparse.mm(S, lin(F.dropout(h, self.dropout, self.training)))    # transform, then average over neighbors
            if li < len(self.lins) - 1:
                h = F.relu(h)
                if self.residual and h.shape == h_in.shape:
                    h = h + h_in
            if return_hidden and li == len(self.lins) - 2:
                hidden = h
        return (h, hidden) if return_hidden else h


class MLP(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.5), nn.Linear(i, h), nn.ReLU(), nn.Dropout(0.5), nn.Linear(h, o))

    def forward(self, X, S=None):
        return self.net(X)


class SAGE(nn.Module):
    """GraphSAGE with a mean aggregator: h' = W_self h + W_neigh mean(neighbors' h). The node's own features are
    kept separate from the neighborhood's, instead of being averaged in with them."""
    def __init__(self, i, h, o):
        super().__init__()
        self.s1, self.n1, self.s2, self.n2 = nn.Linear(i, h), nn.Linear(i, h, bias=False), nn.Linear(h, o), nn.Linear(h, o, bias=False)

    def forward(self, X, M):
        h = F.relu(self.s1(F.dropout(X, 0.5, self.training)) + self.n1(torch.sparse.mm(M, X)))
        h = F.dropout(h, 0.5, self.training)
        return self.s2(h) + self.n2(torch.sparse.mm(M, h))


def mean_adj(A):
    return (A / A.sum(1, keepdim=True).clamp_min(1)).to_sparse()


def fit(model, X, S, y, train_mask, epochs=200, lr=0.01, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    for _ in range(epochs):
        model.train()
        loss = F.cross_entropy(model(X, S)[train_mask], y[train_mask])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return (model(X, S).argmax(1) == y)[~train_mask].float().mean().item()


def label_propagation(A, y, train_mask, k=4, iters=50, alpha=0.9):
    """Graph only: spread the known labels along edges (Zhou et al., 2004)."""
    S = norm_adj(A).to_dense()
    Y0 = torch.zeros(len(y), k); Y0[train_mask, y[train_mask]] = 1
    F_ = Y0.clone()
    for _ in range(iters):
        F_ = alpha * S @ F_ + (1 - alpha) * Y0
    return (F_.argmax(1) == y)[~train_mask].float().mean().item()


S = norm_adj(A)
res = {"MLP (features only)": fit(MLP(16, 64, 4), X, None, y, train_mask),
       "label propagation (graph only)": label_propagation(A, y, train_mask),
       "GCN, 2 layers (both)": fit(GCN(16, 64, 4), X, S, y, train_mask)}
print("\ntest accuracy (chance 0.25)")
for k_, v in res.items():
    print(f"  {k_:32s} {v:.3f}")
assert res["GCN, 2 layers (both)"] > max(res["MLP (features only)"], res["label propagation (graph only)"]) + 0.03

# %% [markdown]
# ## 3. Permutation equivariance

# %%
torch.manual_seed(0)
gcn = GCN(16, 64, 4).eval()
perm = torch.randperm(len(y))
Ap = A[perm][:, perm]
with torch.no_grad():
    out, out_p = gcn(X, S), gcn(X[perm], norm_adj(Ap))
assert torch.allclose(out[perm], out_p, atol=1e-5)
print("\nrenumber the nodes (permute X and the rows and columns of A): the outputs are the same, renumbered")

# %% [markdown]
# ## 4. Over-smoothing
#
# Every GCN layer averages each node with its neighbors. Stack enough of them and every node in a connected component
# converges toward the same vector. Measured: test accuracy, and the mean cosine similarity between the hidden vectors
# of random pairs of nodes (1 means every node looks the same).

# %%
def mean_pair_cosine(H, n_pairs=5000, seed=0):
    g = torch.Generator().manual_seed(seed)
    i, j = torch.randint(0, len(H), (2, n_pairs), generator=g)
    return F.cosine_similarity(H[i], H[j], dim=1).mean().item()


print("\nlayers   accuracy   pair cosine   | with residuals: accuracy   pair cosine")
smooth = {}
t0 = time.time()
for L in (2, 4, 8, 16, 32):
    row = []
    for residual in (False, True):
        m = GCN(16, 64, 4, layers=L, residual=residual, dropout=0.5 if L <= 4 else 0.1)
        acc = fit(m, X, S, y, train_mask, epochs=300)
        with torch.no_grad():
            _, H = m.eval()(X, S, return_hidden=True)
        row += [acc, mean_pair_cosine(H)]
    smooth[L] = row
    print(f"{L:6d}   {row[0]:8.3f}   {row[1]:11.3f}   |                  {row[2]:8.3f}   {row[3]:11.3f}")
print(f"({time.time() - t0:.0f}s) without residuals, depth makes every node's vector the same and accuracy falls to")
print("guessing. Residual connections keep each node's own signal alive (26.4 again).")
assert smooth[32][0] < smooth[2][0] - 0.3 and smooth[32][1] > 0.9
assert smooth[32][2] > smooth[32][0] + 0.2

# %% [markdown]
# ## 5. Heterophily
#
# Same features, but now edges connect *different* classes far more often than the same class.

# %%
A_het, X_het, y_het = sbm(p_in=0.005, p_out=0.03, seed=1)
m_het = split(y_het)
hom = (A_het * (y_het[:, None] == y_het[None, :])).sum() / A_het.sum()
res_het = {"MLP (features only)": fit(MLP(16, 64, 4), X_het, None, y_het, m_het),
           "GCN": fit(GCN(16, 64, 4), X_het, norm_adj(A_het), y_het, m_het),
           "GraphSAGE (mean)": fit(SAGE(16, 64, 4), X_het, mean_adj(A_het), y_het, m_het)}
print(f"\nheterophilic graph: edge homophily {hom:.2f}")
for k_, v in res_het.items():
    print(f"  {k_:20s} {v:.3f}")
print("the GCN averages a node with neighbors that are mostly other classes and loses to a model that ignores the graph.")
print("GraphSAGE keeps the node's own features in a separate channel and can learn that the neighborhood means 'not me'.")
assert res_het["GCN"] < res_het["MLP (features only)"]
assert res_het["GraphSAGE (mean)"] > res_het["GCN"] + 0.05

print("\nAll checks passed.")
