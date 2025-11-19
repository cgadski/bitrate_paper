# %%
%load_ext autoreload
%autoreload 2

# %%
from project.pursuit import pursuit, rademacher, DTYPE
from project.misc import step_sizes
import torch as t

n = 1024
f = rademacher((n, 128)).to(dtype=DTYPE)
weights = t.ones(256, 1024)
k = 1
d = 10
pursuit(f, weights, d, 1, 1)


# %%
import vandc
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from project.graphs.capacity import CapacityGraph

# %%
df = vandc.collate_runs(vandc.fetch_dir(Path("../results/eta_sweep/")))

# %%
data = df[(df["n"] == 2**20) & (df["max_steps"] == 4) & (df["threshold"] == False)].groupby(["eta", "factor"]).mean("acc").reset_index()
CapacityGraph(data).plot()
