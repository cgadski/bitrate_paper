"""
For a given number N of codewords, runs experiments
"""

from dataclasses import dataclass
from math import ceil, exp, log
from typing import Any, Literal

import torch as t
import vandc
from loguru import logger
from simple_parsing import parse

from project.misc import grid

from .sparse_recovery import DTYPE, map_threshold, matching_pursuit, rademacher


def binom_entropy(n, k):
    return k * (1 + log(n) - log(k))


@dataclass
class Options:
    n: int
    dict_type: Literal["spherical", "rademacher"] = "rademacher"

    max_d_per_nat: float = 9
    max_eta: float = 1 / 2

    batch: int = 64
    resolution: int = 64
    max_floats: int = 5_000_000
    device: str = "cpu"

    def max_d(self):
        max_nats = binom_entropy(self.n, exp(self.max_eta * log(self.n)))
        ideal_d = int(self.max_d_per_nat * max_nats)
        return min(int(self.max_floats / self.n), ideal_d)


def go(opts: Options):
    vandc.init(opts)

    t.set_default_device(opts.device)

    if opts.dict_type == "spherical":
        f = t.randn((opts.n, opts.max_d()), dtype=DTYPE)
    else:
        f = rademacher((opts.n, opts.max_d())).to(dtype=DTYPE)
    weights = t.ones(opts.batch, opts.n)

    logger.info(f"Created dictionary of shape {f.shape}")

    eta = t.linspace(0, opts.max_eta, opts.resolution)
    d_per_nat = t.linspace(0, opts.max_d_per_nat, opts.resolution)

    for params in vandc.progress(list(grid(eta=eta, d_per_nat=d_per_nat))):
        k = int(exp(params["eta"] * log(opts.n)))
        nats = binom_entropy(opts.n, k)
        d = max(1, int(params["d_per_nat"] * nats))
        if d > opts.max_d():
            continue

        def on_record(record: dict[str, Any], method: str):
            record["eta"] = params["eta"]
            record["d_per_nat"] = params["d_per_nat"]
            record["method"] = method
            vandc.log(record)

        on_record(map_threshold(f, weights, d, k), "map")
        on_record(matching_pursuit(f, weights, d, k, 1), "top_k")
        on_record(matching_pursuit(f, weights, d, k, 2), "2_step")
        on_record(matching_pursuit(f, weights, d, k, 3), "3_step")
        on_record(matching_pursuit(f, weights, d, k, 64), "64_step")

    vandc.close()


if __name__ == "__main__":
    opts = parse(Options)
    go(opts)
