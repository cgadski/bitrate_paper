# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import vandc
from vandc.writer import git_root

from project.graphs.eta_sweep import CapacityGraph

# %%
df = vandc.collate_runs(vandc.fetch_dir(Path("../results") / "eta_sweep_2"))

# %%
values = (
    df.groupby(["n", "k", "d", "threshold", "max_steps"])
    .mean("acc")
    .reset_index()[["n", "k", "d", "threshold", "max_steps", "acc"]]
)


# %%


def average_results(df):
    cells = (
        df[["dict_type", "n", "eta", "d_per_nat", "threshold", "max_steps", "k", "d"]]
        .drop_duplicates()
        .sort_values(["dict_type", "n", "eta", "d_per_nat", "method"])
    )  # pyright: ignore

    values = (
        df.groupby(["n", "k", "d", "threshold", "max_steps"])
        .mean("acc")
        .reset_index()[["n", "k", "d", "threshold", "max_steps", "acc"]]
    )
    values = values.set_index(["n", "k", "d", "method"])
    data = cells.join(
        values, on=["n", "k", "d", "threshold", "max_steps"], validate="m:1"
    )
    return data


if __name__ == "__main__":
    runs = list(vandc.fetch_dir(Path("results") / "eta_sweep_2"))
    print(f"Found {len(runs)} runs")
    df = average_results(vandc.collate_runs(runs))
    # df.to_csv(rot / "results" / "eta_sweep_2.csv", index=False)
    print(df)

    # fig, ax = plt.subplots()
    # CapacityGraph(df).make_subplot(ax, 2**20, 3)
