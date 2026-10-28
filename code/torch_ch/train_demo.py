import numpy as np
from mini_torch import MiniLinear, MiniSGD, MiniTensor

rng = np.random.default_rng(0)
X = np.linspace(-1, 1, 50).reshape(-1, 1)
Y = 3 * X - 2                                    # the line we want to recover

model = MiniLinear(1, 1, rng)
opt = MiniSGD(model.parameters(), lr=0.5)

for step in range(60):
    model.zero_grad()                            # 1. forget the previous gradients
    pred = model(MiniTensor(X))                  # 2. forward: compute, and record the tape
    loss = ((pred - MiniTensor(Y)) ** 2).mean()  # 3. a single number to minimize
    loss.backward()                              # 4. replay the tape: fill every .grad
    opt.step()                                   # 5. update the parameters
    if step in (0, 9, 29, 59):
        print(f"step {step:2d}  loss {float(loss.data):.2e}")

print(model.weight.data.round(4), model.bias.data.round(4))
