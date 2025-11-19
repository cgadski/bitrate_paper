import torch as t
from math import log

DTYPE = t.float16


def step_sizes(max_len, sum):
    remainder = sum
    while remainder > 0:
        step = -(remainder // -max_len)
        remainder -= step
        max_len -= 1
        yield step


def rademacher(shape):
    return t.where(t.randn(shape) > 0, 1, -1)


def matching_pursuit(f, weights, d, k, max_steps):
    signal = t.multinomial(weights, k)  # b k -> n
    code = f[signal, :d].sum(dim=1, dtype=DTYPE)  # b n
    predicted = t.zeros_like(signal)
    residual = code

    progress = 0
    for step in step_sizes(max_steps, k):
        to_add = t.topk(residual @ f.T[:d], dim=1, k=step).indices
        # to_add: b step -> n
        predicted[:, progress : progress + step] = to_add
        residual -= f[to_add, :d].sum(dim=1)
        progress += step

    signal = signal.sort().values
    predicted = predicted.sort().values
    acc = ((signal == predicted).sum(dim=-1) == k).mean(dtype=t.float)

    return {"k": k, "d": d, "acc": acc}


def map_threshold(f, weights, d, k):
    signal = t.multinomial(weights, k)  # b k -> n
    code = f[signal, :d].sum(dim=1, dtype=DTYPE)  # b n

    b = signal.shape[0]
    n = f.shape[0]
    prec = k / d
    eps = k / n

    tau = 1 / 2 - prec * (log(eps) - log(1 - eps))
    errors = (code @ f.T[:d] > d * tau)[t.arange(b)[:, None], signal]  # b k
    acc = (errors.sum(dim=-1) == k).mean(dtype=t.float)

    return {"k": k, "d": d, "acc": acc}
