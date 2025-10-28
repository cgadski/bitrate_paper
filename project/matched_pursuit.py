from dataclasses import dataclass
from project.misc import grid, random_k, step_sizes
from simple_parsing import parse
import vandc
import torch as t
from math import ceil


def rademacher(shape):
    return t.where(t.randn(shape) > 0, 1, -1)


@dataclass
class Options:
    n: int
    k_step: int  # 128 / 64 = 2
    d_step: int  # 2048 / 64 = 2^(11 - 6) = 2^5 = 32
    max_steps: int = 1

    batch: int = 64
    resolution = 64
    device: str = "cpu"

    def max_d(self):
        return self.d_step * self.resolution

    def step_sizes(self, k):
        return step_sizes(self.max_steps, k)


DTYPE = t.float16


class MPExperiment:
    def __init__(self, opts: Options):
        self.opts = opts
        self.f = rademacher((opts.n, opts.max_d())).to(dtype=DTYPE)
        self.weights = t.ones(opts.batch, opts.n)

    def run(self, k, d):
        o = self.opts

        signal = t.multinomial(self.weights, k)  # b k -> n
        code = self.f[signal, :d].sum(dim=1, dtype=DTYPE)  # b n
        predicted = t.zeros_like(signal)
        residual = code

        progress = 0
        for step in o.step_sizes(k):
            to_add = t.topk(residual @ self.f.T[:d], dim=1, k=step).indices
            # to_add: b step -> n
            predicted[:, progress : progress + step] = to_add
            residual -= self.f[to_add, :d].sum(dim=1)
            progress += step

        signal = signal.sort().values
        predicted = predicted.sort().values
        acc = ((signal == predicted).sum(dim=-1) == k).mean(dtype=t.float)

        return {"k": k, "d": d, "acc": acc}


if __name__ == "__main__":
    opts = parse(Options)
    vandc.init(opts)

    t.set_default_device(opts.device)

    experiment = MPExperiment(opts)

    k = t.arange(1, opts.resolution + 1) * opts.k_step
    d = t.arange(1, opts.resolution + 1) * opts.d_step

    for args in vandc.progress(list(grid(k=k, d=d))):
        vandc.log(experiment.run(**args))
