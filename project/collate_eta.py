# %%
from vandc.writer import git_root
import vandc
from pathlib import Path
import pandas as pd
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


# %%
root: Path = git_root()  # pyright: ignore
runs = list(vandc.fetch_dir(root / "results" / "eta_sweep_mid"))

df = average_results(vandc.collate_runs(runs))

df.to_csv(root / "results" / "eta_sweep_mid.csv", index=False)

fig, ax = plt.subplots()
CapacityGraph(df).make_subplot(ax, 2 ** 20, 3)
