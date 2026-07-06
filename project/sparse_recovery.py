from math import log

import torch as t

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


def matching_pursuit(f, weights, d, k, steps):
    signal = t.multinomial(weights, k)  # b k -> n

    word_mags = f[:, :d].pow(2).sum(dim=1, dtype=DTYPE).sqrt()  # n
    code = (f[signal, :d] / word_mags[signal, None]).sum(dim=1, dtype=DTYPE)  # b d

    predicted = t.zeros_like(signal)  # b k -> n
    residual = code  # b d

    progress = 0
    for step in step_sizes(steps, k):
        top_words = t.topk((residual @ f.T[:d]) / word_mags, dim=1, k=step).indices
        # top_words: b step -> n
        predicted[:, progress : progress + step] = top_words
        residual -= (f[top_words, :d] / word_mags[top_words, None]).sum(dim=1)
        progress += step

    signal = signal.sort().values
    predicted = predicted.sort().values
    acc = ((signal == predicted).sum(dim=-1) == k).mean(dtype=t.float)

    return {"k": k, "d": d, "acc": acc}


def map_threshold(f, weights, d, k):
    signal = t.multinomial(weights, k)  # b k -> n

    word_mags = f[:, :d].pow(2).sum(dim=1, dtype=DTYPE).sqrt()  # n
    code = (f[signal, :d] / word_mags[signal, None]).sum(dim=1, dtype=DTYPE)  # b d

    b = signal.shape[0]
    n = f.shape[0]
    eps = k / n
    var = eps * (k - 1) / d + (1 - eps) * k / d

    tau = 1 / 2 - var * (log(eps) - log(1 - eps))
    recovered = (code @ f.T[:d]) / word_mags > tau  # b n
    total_positive = recovered.sum(dim=-1)
    true_positive = recovered[t.arange(b)[:, None], signal].sum(dim=-1)
    acc = ((total_positive == k) & (true_positive == k)).mean(dtype=t.float)

    return {"k": k, "d": d, "acc": acc}
