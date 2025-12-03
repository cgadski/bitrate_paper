from dataclasses import dataclass
from project.misc import grid
from .sparse_recovery import map_threshold, rademacher, matching_pursuit, DTYPE, gabor_frame
from simple_parsing import parse
import vandc
import torch
from math import ceil, log
from loguru import logger


@dataclass
class Options:
    k_step: int = 1  # 64 / 64 = 1
    t_step: int = 1  # 1024 / 64 = 2^(10 - 6) = 2^4 = 16

    batch: int = 128
    resolution = 32
    device: str = "cpu"


def energy(frame):
    # frame: n d
    n, _ = frame.shape
    total = (frame.T @ frame).pow(2).sum() - n
    return total / (n * (n - 1))

def go_t(opts:Options, t:int):
    n = t * t
    d = 2 * t

    weights = torch.ones(opts.batch, n)
    f_spherical = torch.randn((n, d)).to(dtype=DTYPE)
    f_spherical = f_spherical / f_spherical.norm(dim=1, keepdim=True)
    f_gabor = gabor_frame(t).to(dtype=DTYPE)

    energy_spherical = energy(f_spherical)
    energy_gabor = energy(f_gabor)

    def on_record(record, type, method):
        record["n"] = n
        record["type"] = type
        record["method"] = method
        if type == "spherical":
            record["energy"] = energy_spherical
        else:
            record["energy"] = energy_gabor
        vandc.log(record)

    for i in range(1, opts.resolution + 1):
        k = opts.k_step * i
        if k > n:
            break

        record = matching_pursuit(f_spherical, weights, d, k, 1)
        on_record(record, "spherical", "topk")

        record = matching_pursuit(f_spherical, weights, d, k, 3)
        on_record(record, "spherical", "gmp")

        record = matching_pursuit(f_gabor, weights, d, k, 1)
        on_record(record, "gabor", "topk")

        record = matching_pursuit(f_gabor, weights, d, k, 3)
        on_record(record, "gabor", "gmp")


def go(opts: Options):
    vandc.init(opts)

    torch.set_default_device(opts.device)

    for i in vandc.progress(range(1, opts.resolution + 1)):
        go_t(opts, opts.t_step * i)

    vandc.close()


if __name__ == "__main__":
    opts = parse(Options)
    go(opts)
