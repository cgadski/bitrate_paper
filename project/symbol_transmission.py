import torch as t
from math import ceil, sqrt, log
import matplotlib.pyplot as plt
from einops import einsum
import vandc
from simple_parsing import parse
from dataclasses import dataclass
from loguru import logger

from project.misc import grid


@dataclass
class Options:
    noise: float
    n_step: float # 16 / 64 = 0.25
    d_step: int # 2^10 / 2^6 = 16
    batch: int

    frame: str = "spherical"
    resolution: int = 64

    def max_n(self):
        return 2 ** ceil(self.n_step * self.resolution)

    def max_d(self):
        return self.d_step * self.resolution


def rademacher(shape):
    return t.where(t.randn(*shape) > 0, 1, -1)


class Experiment:
    def __init__(self, opts: Options):
        self.opts = opts

        self.f = rademacher((opts.max_n(), opts.max_d())).to(t.float)

        logger.info(f"Initialized dictionary with dimensions {(opts.max_n(), opts.max_d())}")


    def measure(self, n, d):
        opts = self.opts

        z = t.randn(opts.batch, d)
        z *= sqrt(opts.noise)
        s = t.randint(n, (opts.batch,))

        transmission = self.f[s, :d] + z  # b d
        filters = transmission @ self.f.T[:d, :n]  # b n

        max_decoded = t.max(filters, dim=1).indices
        max_correct = (max_decoded == s).mean(dtype=t.float).item()

        return {
            "noise": opts.noise,
            "n": n,
            "d": d,
            "success": max_correct,
        }


def run(opts: Options):
    range = t.arange(0, opts.resolution) + 1
    args = {
        "d": range * opts.d_step,
        "n": (2 ** (range * opts.n_step)).to(t.int).unique(),
    }

    e = Experiment(opts)

    for arg in vandc.progress(list(grid(**args))):
        if arg["n"] == 1:
            pass
        vandc.log(e.measure(**arg))


if __name__ == "__main__":
    opts = parse(Options)
    vandc.init(opts)
    run(opts)
    vandc.close()
