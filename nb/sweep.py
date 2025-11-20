# %%
%load_ext autoreload
%autoreload 2

# %%
from project.sparse_recovery import map_threshold, matching_pursuit, rademacher, DTYPE
import torch as t

n = 1024
# f = rademacher((n, 128)).to(dtype=DTYPE)
f = t.randn((n, 128), dtype=DTYPE)
weights = t.ones(256, 1024)
k = 1
d = 5
matching_pursuit(f, weights, d, 1, 1)
# %%
import vandc
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from project.graphs.eta_sweep import CapacityGraph
import matplotlib.pyplot as plt

# %%

def average_results(df):
    cells = (
        df[["n", "eta", "factor", "threshold", "max_steps", "k", "d"]]
        .drop_duplicates()
        .sort_values(["n", "eta", "factor", "threshold", "max_steps"])
    )  # pyright: ignore

    # %%
    values = (
        df.groupby(["n", "k", "d", "threshold", "max_steps"])
        .mean("acc")
        .reset_index()[["n", "k", "d", "threshold", "max_steps", "acc"]]
    )
    values = values.set_index(["n", "k", "d", "threshold", "max_steps"])
    data = cells.join(values, on=["n", "k", "d", "threshold", "max_steps"], validate="m:1")
    return data

runs = vandc.fetch_all()
df = average_results(vandc.collate_runs(runs))

fig, axs = plt.subplots(5)
CapacityGraph(df).make_subplot(ax, 2 ** 8, 'threshold')

# %%
data = df[(df["n"] == 2**20) & (df["max_steps"] == 4) & (df["threshold"] == False)].groupby(["eta", "factor"]).mean("acc").reset_index()
CapacityGraph(data).plot()
