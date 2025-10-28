# %%
from math import sqrt, log
import torch as t
import numpy as np
from dataclasses import dataclass
from .misc import random_k


# %%
def gabor(seed: t.Tensor):
    d = seed.shape[0]
    idx = t.arange(d)
    theta = 2 * t.pi * t.arange(d) / d
    m = t.complex(t.cos(theta), t.sin(theta))
    return (
        seed[None, (idx[:, None] + idx[None, :]) % d]
        * m[(idx[:, None, None] * idx[None, None, :]) % d]
    )  # mod trans d


def gabor_frame(d):
    seed = t.complex(t.randn(d), t.randn(d))
    frame = gabor(seed).flatten(0, 1)
    out = t.zeros(d * d, 2 * d)
    out[:, :d] = frame.real
    out[:, d:] = frame.imag
    out /= out.norm(dim=1, keepdim=True)
    return out


def rademacher(shape):
    return t.where(t.randn(*shape) > 0, 1, -1)


def rademacher_frame(n, d):
    return (1 / sqrt(d)) * rademacher((n, d)).to(dtype=t.float)


def energy(frame):
    # frame: n d
    n, _ = frame.shape
    total = (frame.T @ frame).pow(2).sum() - n
    return t.sqrt(total / (n * (n - 1)))


d = 10
print(d * d)
print(energy(gabor_frame(d)))
print(energy(rademacher_frame(d * d, 2 * d)))
rademacher_frame(d * d, 2 * d).shape
# frame @ frame.T


# %%
@dataclass
class LinearRecovery:
    n: int
    d: int
    b: int = 64
    use_gabor: bool = False

    def __post_init__(self):
        if self.use_gabor:
            self.frame = gabor_frame(int(self.d / 2))
            self.n = self.frame.shape[0]
            self.d = self.frame.shape[1]
        else:
            self.frame = rademacher_frame(self.n, self.d)

    def experiment(self, k):
        signals = random_k(n=self.n, k=k, b=self.b)  # b n
        projections = signals @ self.frame
        filtered = projections @ self.frame.T  # b n

        thresholded = t.where(filtered > 1 / 2, 1, 0)  # b n
        topk_indices = t.topk(filtered, k=k).indices  # b k -> n
        threshold_errors = self.n - t.sum(thresholded == signals, dim=1)
        topk_errors = 2 * (
            k - t.sum(signals[t.arange(self.b)[:, None], topk_indices], dim=1)
        )

        return {
            "threshold_accuracy": (threshold_errors == 0).mean(dtype=t.float).item(),
            "topk_accuracy": (topk_errors == 0).mean(dtype=t.float).item(),
        }


# %%
from math import log

d_half = 400
d = 2 * d_half
n = d_half * d_half
min_k = (1 / 8) * d / log(n)
print(f"d: {d}, n: {n}, min_k: {min_k}")
e = LinearRecovery(n=n, d=d, b=64, use_gabor=False)
e.experiment(17)
