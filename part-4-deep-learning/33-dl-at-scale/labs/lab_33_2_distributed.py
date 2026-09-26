# %% [markdown]
# # Lab 33.2: Distributed training, on two CPU processes
#
# Real torch.distributed with the gloo backend: two processes on this machine stand in for two GPUs. Everything is
# checked against a single-process reference, because the whole point of these techniques is that the math doesn't change.
#
# 1. Data parallelism by hand: all-reduce the gradients. Then DDP, which does the same with bucketing and overlap.
# 2. ZeRO-1 by hand: shard the optimizer state, all-gather the updated parameters.
# 3. FSDP2 (fully_shard): parameters, gradients and optimizer state sharded; same training trajectory.
# 4. Tensor parallelism: a Megatron-style MLP split across the two processes, one all-reduce per forward.
# 5. Measuring all-reduce: latency and bandwidth (the alpha-beta model).
# 6. Pipeline parallelism: the bubble, computed and drawn.

# %%
import os

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")                  # quiet c10d's IPv6 warnings on some machines
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

WORLD = 2


def make_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 10))


def make_data(step, n=64):
    g = torch.Generator().manual_seed(1000 + step)
    return torch.randn(n, 32, generator=g), torch.randint(0, 10, (n,), generator=g)


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def worker(rank, port):
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    torch.set_num_threads(1)
    shard = lambda t: t.chunk(WORLD)[rank]                                   # this rank's half of a batch

    # reference: one process, the full batch
    ref = make_model()
    x, y = make_data(0)
    F.cross_entropy(ref(x), y).backward()
    ref_grads = [p.grad.clone() for p in ref.parameters()]

    # 1. data parallelism
    m = make_model()
    F.cross_entropy(m(shard(x)), shard(y)).backward()                        # each rank: its half of the batch
    for p in m.parameters():
        dist.all_reduce(p.grad)                                              # sum over ranks...
        p.grad /= WORLD                                                      # ...then average
    err = max((p.grad - r).abs().max().item() for p, r in zip(m.parameters(), ref_grads))
    log(rank, f"1. all-reduced gradients from two half-batches vs one full batch: max difference {err:.1e}")
    assert err < 1e-6

    ddp = nn.parallel.DistributedDataParallel(make_model())
    F.cross_entropy(ddp(shard(x)), shard(y)).backward()
    err = max((p.grad - r).abs().max().item() for p, r in zip(ddp.module.parameters(), ref_grads))
    log(rank, f"   DDP (same thing, bucketed and overlapped with the backward pass): max difference {err:.1e}")
    assert err < 1e-6

    # 2. ZeRO-1: each rank keeps Adam state for its half of the parameters only.
    # The reference is torch's Adam on the full parameter vector, fed the same all-reduced gradient. Section 1 already
    # showed that gradient equals the full-batch one up to rounding (~1e-8); feeding the reference a separately computed
    # full-batch gradient would test that again, badly, because Adam divides each coordinate by its own running scale:
    # where a gradient is ~1e-8, rounding noise of ~1e-8 changes the step by a visible fraction of the learning rate,
    # and how visible depends on the CPU. What ZeRO claims is narrower: sharding the optimizer state changes nothing.
    ref = make_model()
    ref_opt = torch.optim.Adam(ref.parameters(), lr=1e-2)
    m = make_model()
    flat = torch.cat([p.data.flatten() for p in m.parameters()])
    lo, hi = rank * len(flat) // WORLD, (rank + 1) * len(flat) // WORLD
    my = flat[lo:hi].clone()
    exp_avg, exp_sq = torch.zeros_like(my), torch.zeros_like(my)             # optimizer state: this rank's shard only
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 1e-2
    for step in range(1, 4):
        xs, ys = make_data(step)
        for p in m.parameters():
            p.grad = None
        F.cross_entropy(m(shard(xs)), shard(ys)).backward()
        g = torch.cat([p.grad.flatten() for p in m.parameters()])
        dist.all_reduce(g); g /= WORLD                                       # (real ZeRO uses reduce-scatter: only the shard)
        i = 0
        for r in ref.parameters():                                           # the reference: same gradient, unsharded Adam
            r.grad = g[i:i + r.numel()].view_as(r).clone(); i += r.numel()
        ref_opt.step()
        g = g[lo:hi]
        exp_avg.mul_(b1).add_(g, alpha=1 - b1); exp_sq.mul_(b2).addcmul_(g, g, value=1 - b2)
        my -= lr * (exp_avg / (1 - b1 ** step)) / ((exp_sq / (1 - b2 ** step)).sqrt() + eps)
        parts = [torch.empty((r + 1) * len(flat) // WORLD - r * len(flat) // WORLD) for r in range(WORLD)]
        dist.all_gather(parts, my)                                           # everyone gets everyone's updated shard
        new = torch.cat(parts)
        i = 0
        for p in m.parameters():
            p.data.copy_(new[i:i + p.numel()].view_as(p)); i += p.numel()
    err = max((p - r).abs().max().item() for p, r in zip(m.parameters(), ref.parameters()))
    n_params = len(flat)
    log(rank, f"\n2. ZeRO-1, 3 Adam steps: max parameter difference from unsharded Adam on the same gradients {err:.1e}")
    log(rank, f"   Adam state per rank: {2 * len(my):,} numbers instead of {2 * n_params:,}")
    assert err < 1e-6

    # 3. FSDP2
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    mesh = init_device_mesh("cpu", (WORLD,))
    ref = make_model()
    ref_opt = torch.optim.Adam(ref.parameters(), lr=1e-2)
    fs = make_model()
    for layer in fs:
        if isinstance(layer, nn.Linear):
            fully_shard(layer, mesh=mesh)                                     # each Linear is a unit: gathered just in time
    fully_shard(fs, mesh=mesh)
    opt = torch.optim.Adam(fs.parameters(), lr=1e-2)
    grad_err = 0.0
    for step in range(1, 4):
        xs, ys = make_data(step)
        opt.zero_grad(); F.cross_entropy(fs(shard(xs)), shard(ys)).backward()
        full_grads = [p.grad.full_tensor() for p in fs.parameters()]         # gather the sharded gradients, to check them
        check = make_model()                                                 # full batch, one process, at the same parameters
        with torch.no_grad():
            for c, p in zip(check.parameters(), fs.parameters()):
                c.copy_(p.full_tensor())
        F.cross_entropy(check(xs), ys).backward()
        grad_err = max(grad_err, max((g - c.grad).abs().max().item() for g, c in zip(full_grads, check.parameters())))
        for r, g in zip(ref.parameters(), full_grads):                       # reference: same gradients, unsharded Adam
            r.grad = g.clone()
        ref_opt.step(); opt.step()
    local = sum(p.to_local().numel() for p in fs.parameters())
    err = max((p.full_tensor() - r).abs().max().item() for p, r in zip(fs.parameters(), ref.parameters()))
    log(rank, f"\n3. FSDP2, 3 Adam steps: sharded gradients vs one process on the full batch, max difference {grad_err:.1e};")
    log(rank, f"   parameters vs unsharded Adam on the same gradients, max difference {err:.1e}")
    log(rank, f"   parameters stored on this rank: {local:,} of {n_params:,} (and gradients and Adam state likewise)")
    assert grad_err < 1e-6 and err < 1e-6 and local <= n_params // WORLD + 128

    # 4. tensor parallelism: y = W2 gelu(W1 x). Split W1 by output rows, W2 by input columns.
    torch.manual_seed(0)
    d = 64
    W1, W2 = torch.randn(4 * d, d) / d ** 0.5, torch.randn(d, 4 * d) / (4 * d) ** 0.5
    xt = torch.randn(8, d)
    full = F.gelu(xt @ W1.T) @ W2.T
    W1_r, W2_r = W1.chunk(WORLD, 0)[rank], W2.chunk(WORLD, 1)[rank]         # column-parallel, then row-parallel
    partial = F.gelu(xt @ W1_r.T) @ W2_r.T                                  # no communication until here: gelu is per element
    dist.all_reduce(partial)                                                 # one all-reduce of an (8 x d) activation
    err = (partial - full).abs().max().item()
    log(rank, f"\n4. tensor-parallel MLP (each rank holds half of W1 and W2): max difference from the full MLP {err:.1e}")
    log(rank, f"   communication per forward: one all-reduce of {partial.numel():,} numbers; weights per rank: {W1_r.numel() + W2_r.numel():,} of {W1.numel() + W2.numel():,}")
    assert err < 1e-5

    # 5. all-reduce timing
    if rank == 0:
        print("\n5. all-reduce over gloo on one machine (loopback): time vs message size", flush=True)
    sizes = [2 ** k for k in (4, 10, 14, 18, 21, 23)]
    times = []
    for n in sizes:
        t = torch.ones(n)
        for _ in range(3):
            dist.all_reduce(t)
        dist.barrier()
        t0 = time.perf_counter()
        reps = 10
        for _ in range(reps):
            dist.all_reduce(t)
        times.append((time.perf_counter() - t0) / reps)
    if rank == 0:
        for n, tt in zip(sizes, times):
            print(f"   {n * 4:>12,} bytes   {tt * 1e6:9.0f} us   {n * 4 / tt / 1e9:6.2f} GB/s", flush=True)
        print("   small messages cost a fixed latency (alpha), large ones are limited by bandwidth (beta): batch your communication.")
        assert times[0] > 0 and sizes[-1] * 4 / times[-1] > 10 * sizes[0] * 4 / times[0]
    dist.destroy_process_group()


def pipeline_bubble(p, m):
    """GPipe: p stages, m micro-batches, forward then backward. Each cell is one time slot; backward takes 2 slots."""
    rows = []
    for s in range(p):
        fwd = " " * s + "".join(str(i % 10) for i in range(m)) + " " * (p - 1 - s)
        bwd = " " * (2 * (p - 1 - s)) + "".join(str(i % 10) * 2 for i in reversed(range(m))) + " " * (2 * s)
        rows.append(f"   stage {s}: |{fwd}|{bwd}|")
    idle = (p - 1) / (m + p - 1)
    return rows, idle


if __name__ == "__main__":
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    t0 = time.time()
    mp.spawn(worker, args=(port,), nprocs=WORLD)

    print("\n6. pipeline parallelism, GPipe schedule, 4 stages (forward digits, then backward at twice the width):")
    for m_ in (4, 16):
        rows, idle = pipeline_bubble(4, m_)
        print(f"   {m_} micro-batches: idle fraction (the bubble) (p - 1) / (m + p - 1) = {idle:.2f}")
        if m_ == 4:
            print("\n".join(rows))
    assert abs(pipeline_bubble(4, 4)[1] - 3 / 7) < 1e-9
    print(f"\nAll checks passed. ({time.time() - t0:.0f}s)")
