from dataclasses import dataclass

from loguru import logger
from project.misc import grid, random_k, step_sizes
from simple_parsing import parse
import vandc
import torch as t
from math import ceil, log, exp

from .pursuit import pursuit, rademacher, DTYPE


def entropy(n, k):
    return k * (1 + log(n) - log(k))


@dataclass
class Options:
    n: int
    max_factor: float
    max_eta: float = 1 / 2
    max_steps: int = 1

    batch: int = 64
    resolution = 64
    device: str = "cpu"

    def max_d(self):
        max_nats = entropy(self.n, exp(self.max_eta * log(self.n)))
        return int(self.max_factor * max_nats)


def go(opts: Options):
    vandc.init(opts)

    f = rademacher((opts.n, opts.max_d())).to(dtype=DTYPE)
    weights = t.ones(opts.batch, opts.n)

    logger.info(f"Created dictionary of shape {f.shape}")

    t.set_default_device(opts.device)
    eta = t.linspace(0, opts.max_eta, opts.resolution)
    factor = t.linspace(0, opts.max_factor, opts.resolution)

    for args in vandc.progress(list(grid(eta=eta, factor=factor))):
        k = int(exp(args["eta"] * log(opts.n)))
        nats = entropy(opts.n, k)
        d = int(args["factor"] * nats)
        record = pursuit(f, weights, d, k, opts.max_steps)
        record["eta"] = args["eta"]
        record["factor"] = args["factor"]
        vandc.log(record)

    vandc.close()


if __name__ == "__main__":
    opts = parse(Options)
    go(opts)
