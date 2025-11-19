from dataclasses import dataclass
from project.misc import grid
from .sparse_recovery import map_threshold, rademacher, matching_pursuit, DTYPE
from simple_parsing import parse
import vandc
import torch as t
from math import ceil, log


@dataclass
class Options:
    n: int
    k_step: int  # 64 / 64 = 1
    d_step: int  # 4096 / 64 = 2^(12 - 6) = 2^6 = 64

    batch: int = 128
    resolution = 64
    device: str = "cpu"

    def max_d(self):
        return self.d_step * self.resolution


def go(opts):
    vandc.init(opts)

    t.set_default_device(opts.device)

    f = rademacher((opts.n, opts.max_d())).to(dtype=DTYPE)
    weights = t.ones(opts.batch, opts.n)

    k = t.arange(1, opts.resolution + 1) * opts.k_step
    d = t.arange(1, opts.resolution + 1) * opts.d_step

    for args in vandc.progress(list(grid(k=k, d=d))):
        record = map_threshold(f, weights, args["d"], args["k"])
        record["method"] = "threshold"
        vandc.log(record)

        record = matching_pursuit(f, weights, args["d"], args["k"], 1)
        record["method"] = "top_k"
        vandc.log(record)

    vandc.close()


if __name__ == "__main__":
    opts = parse(Options)
    go(opts)
