# %%
from vandc.writer import git_root
import vandc
from pathlib import Path
import pandas as pd
from project.graphs.capacity import CapacityGraph

root: Path = git_root()  # pyright: ignore
df = vandc.collate_runs(vandc.fetch_dir(root / "results" / "eta_sweep"))
# %%
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
data.to_csv(root / "results" / "eta_sweep.csv", index=False)

# CapacityGraph(
#     data[(data["n"] == 2**16) & (data["threshold"] == False) & (data["max_steps"] == 4)]
# ).plot()
