from itertools import product
from typing import Any, Iterable

import torch as t


def to_scalar(value):
    if t.is_tensor(value) and value.numel() == 1:
        return value.item()
    return value


def grid(**params) -> Iterable[dict[str, Any]]:
    param_names = list(params.keys())
    param_values = [params[name] for name in param_names]
    combinations = list(product(*param_values))

    for combo in combinations:
        processed_kwargs = {
            key: to_scalar(value) for key, value in zip(param_names, combo)
        }
        yield processed_kwargs
