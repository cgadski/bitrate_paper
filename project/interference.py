import numpy as np
import torch as t
from . import db


def scale(x):
    return x.pow(2).mean().sqrt().item()


class Embedding:
    def __init__(self, d, n):
        self.d = d
        self.n = n
        self.emb = t.randn(d, n)
        self.emb /= self.emb.norm(dim=0)
        self.grad = t.zeros_like(self.emb)

    def interference_dist(self):
        cov = e.emb.T @ e.emb  # (n, n)
        i, j = np.arange(self.n)[:, None], np.arange(self.n)[None, :]
        return cov[j > i]

    def interference(self):
        total = (self.emb @ self.emb.T).pow(2).sum() - self.n
        return t.sqrt(total / (self.n * (self.n - 1)))

    # for d=128:
    # n=1 << 10: 1 ms
    # n=1 << 11: 2 ms
    # n=1 << 14: 14 ms
    # n=1 << 16: 52 ms
    # for d=1024:
    # n=1 << 10: 45 ms
    # n=1 << 16: ~2.5 seconds
    def interference_grad(self):
        self.grad = (self.emb @ self.emb.T) @ self.emb - self.emb
        return self.grad

    def optimize(self, eta: float = 1, iters: int = 10):
        grad_scale = self.n / (self.d**1.5)
        losses = np.zeros(iters)
        for i in range(iters):
            grad = self.interference_grad()
            self.emb -= eta * (1 / grad_scale) * grad
            self.emb /= self.emb.norm(dim=0)
            losses[i] = self.interference().item()
        return np.array(losses)


# e = Embedding(d=1024, n=1 << 16)
# %timeit e.interference_grad()


# %%
# n = 1 << 9 = 512
def optimize_interference(
    d: int, n: int, eta: float, iters: int, w: db.Writer, plot: bool = True
):
    e = Embedding(d=d, n=n)
    init_loss = e.interference()
    losses = e.optimize(eta=eta, iters=iters)
    if plot:
        plt.plot(losses)
    w.write(d=d, n=n, init_loss=init_loss, final_loss=e.interference())


# %%


def d_range(n: int):
    values = t.round(2 ** t.arange(4, 10 + 0.5, step=0.1)).to(t.int)
    return values[values <= n]


d_range(1 << 10)

# %%
# n = 1 << 9 diagnostics
with db.MemoryWriter() as w:
    for d in tqdm(d_range(1 << 9)):
        optimize_interference(d, 1 << 9, eta=0.02, iters=30, w=w, plot=True)

# %%
# n = 1 << 9 results
with db.Writer("gd_interference_2") as w:
    for d in tqdm(d_range(1 << 9)):
        optimize_interference(d, 1 << 9, eta=0.02, iters=30, w=w, plot=True)

# %%
# n = 1 << 12 diagnostics
n = 1 << 12
with db.MemoryWriter() as w:
    optimize_interference(d=1 << 4, n=n, eta=0.01, iters=28, w=w, plot=True)
# %%
# n = 1 << 12 results
n = 1 << 12
with db.Writer("gd_interference_2") as w:
    for d in tqdm(d_range(n)):
        optimize_interference(d, n, eta=0.01, iters=28, w=w, plot=False)

# %%
# n = 1 << 6 results
n = 1 << 6
with db.Writer("gd_interference_2") as w:
    for d in tqdm(d_range(n)):
        optimize_interference(d, n, eta=0.04, iters=30, w=w)

# %%
n = 1 << 8
with db.Writer("gd_interference_2") as w:
    for d in tqdm(d_range(n)):
        optimize_interference(d, n, eta=0.02, iters=30, w=w)
