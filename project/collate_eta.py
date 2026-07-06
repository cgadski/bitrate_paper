from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import vandc
from vandc.writer import git_root

from project.graphs.eta_sweep import CapacityGraph


def average_results(df):
    cells = (
        df[["n", "eta", "factor", "threshold", "max_steps", "k", "d"]]
        .drop_duplicates()
        .sort_values(["n", "eta", "factor", "threshold", "max_steps"])
    )  # pyright: ignore

    values = (
        df.groupby(["n", "k", "d", "threshold", "max_steps"])
        .mean("acc")
        .reset_index()[["n", "k", "d", "threshold", "max_steps", "acc"]]
    )
    values = values.set_index(["n", "k", "d", "threshold", "max_steps"])
    data = cells.join(
        values, on=["n", "k", "d", "threshold", "max_steps"], validate="m:1"
    )
    return data


if __name__ == "__main__":
    runs = list(vandc.fetch_all("project/eta_sweep_runner.py %"))
    print(f"Found {len(runs)} runs")
    df = average_results(vandc.collate_runs(runs))
    # df.to_csv(rot / "results" / "eta_sweep_2.csv", index=False)
    print(df)

    # fig, ax = plt.subplots()
    # CapacityGraph(df).make_subplot(ax, 2**20, 3)
