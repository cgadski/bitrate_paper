import torch as t
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from itertools import product


def random_k(n, k, b=1024):
    """
    Generate b random k-hot vectors in n dimensions.
    Returns: b n
    """
    ones = t.multinomial(t.ones(b, n), k)  # b k -> n
    signal = t.zeros((b, n))
    signal[np.arange(b)[:, None], ones] = 1
    return signal


def grad_step(x, eta: float):
    with t.no_grad():
        if isinstance(x, nn.Module):
            params = x.parameters()
        elif isinstance(x, list):
            params = x
        else:
            params = [x]

        for p in params:
            if p.grad is not None:
                p.sub_(eta * p.grad)
                p.grad = None


def to_cpu(tensor):
    return tensor.cpu() if hasattr(tensor, "cpu") else tensor


def to_scalar(value):
    if t.is_tensor(value) and value.numel() == 1:
        return value.item()
    return value


def grid(**params):
    param_names = list(params.keys())
    param_values = [params[name] for name in param_names]
    combinations = list(product(*param_values))

    for combo in combinations:
        processed_kwargs = {
            key: to_scalar(value) for key, value in zip(param_names, combo)
        }
        yield processed_kwargs


def collect_grid(f, progress=False, **params) -> pd.DataFrame:
    param_names = list(params.keys())
    param_values = [params[name] for name in param_names]
    combinations = list(product(*param_values))
    if progress:
        combinations = tqdm(combinations)

    results = []
    for combo in combinations:
        processed_kwargs = {
            key: to_scalar(value) for key, value in zip(param_names, combo)
        }
        result = f(**processed_kwargs)
        processed_result = {key: to_scalar(value) for key, value in result.items()}
        results.append({**processed_kwargs, **processed_result})

    df = pd.DataFrame(results)
    cols = param_names + [col for col in df.columns if col not in param_names]
    df = df[cols]

    return df  # pyright: ignore


def center(x):
    return x - x.mean()
