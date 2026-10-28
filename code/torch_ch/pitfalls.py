import io

import numpy as np
import torch
from torch import nn

torch.manual_seed(0)

print("--- 1. dtype")
X = np.random.default_rng(0).normal(size=(4, 3))
layer = nn.Linear(3, 1)
print(X.dtype, layer.weight.dtype)
try:
    layer(torch.from_numpy(X))
except RuntimeError as e:
    print("RuntimeError:", e)
print(layer(torch.from_numpy(X).float()).shape)

print("--- 2. in-place")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.exp()
y.add_(1)
try:
    y.sum().backward()
except RuntimeError as e:
    print("RuntimeError:", str(e)[:400])
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2
y.add_(1)
y.sum().backward()
print(x.grad)
w = torch.ones(3, requires_grad=True)
try:
    w -= 0.1
except RuntimeError as e:
    print("RuntimeError:", e)
with torch.no_grad():
    w -= 0.1
print(w)

print("--- 3. dropout")
torch.manual_seed(0)
drop = nn.Dropout(p=0.5)
ones = torch.ones(10)
print(drop.training, drop(ones))
drop.eval()
print(drop.training, drop(ones))
net = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 1))
inp = torch.randn(1, 3)
net.train(); a, b = net(inp).item(), net(inp).item()
net.eval(); c, d = net(inp).item(), net(inp).item()
print(round(a, 4), round(b, 4), "|", round(c, 4), round(d, 4))
net.train()
print(all(m.training for m in net.modules()))
net.eval()
print(any(m.training for m in net.modules()))
with torch.no_grad():
    out = net(inp)
print(out.requires_grad, net(inp).requires_grad)

print("--- 4. state_dict")
model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 1))
for k, v in model.state_dict().items():
    print(k, tuple(v.shape))
print(sum(p.numel() for p in model.parameters()), sum(v.numel() for v in model.state_dict().values()))
model.train(); model(torch.randn(16, 3))
buf = io.BytesIO(); torch.save(model.state_dict(), buf); buf.seek(0)
fresh = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 1))
print(fresh.load_state_dict(torch.load(buf)))
model.eval(); fresh.eval()
t = torch.randn(5, 3)
print(torch.equal(model(t), fresh(t)))
wrong = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 1))
try:
    wrong.load_state_dict(model.state_dict())
except RuntimeError as e:
    print("RuntimeError:", str(e)[:600])
