# %%
%load_ext autoreload
%autoreload 2


# %%
import vandc
import matplotlib.pyplot as plt
import torch as t
from pathlib import Path

# %%
runs = vandc.fetch_dir(Path("../server_results"))
runs = [run for run in runs if len(run.logs) > 0]
cols = "k d n".split()
vandc.collate_runs(runs)[cols + ["acc"]].to("results/topk.csv", index=False)

# %%
run = runs[3]
matrix = run.logs.pivot(index="d", columns="k", values="acc")

plt.pcolormesh(
    matrix.columns,  # k values
    matrix.index,  # d values
    matrix,
    shading="nearest",
    rasterized=True,
)

n = t.tensor(run.config["n"])
print(n)

def upper(k):
    eta = t.log(k) / t.log(n)
    c = 2 + 4 * t.sqrt(eta) + 2 * eta
    return c * k * t.log(n)

k = t.linspace(0, 64, 100)
# plt.plot(k, 4 * k * t.log(65536 * k))
plt.plot(k, upper(k))

# %%
runs = [run for run in vandc.fetch_all("%matched_pursuit%") if run.config["device"] == 'cuda']
runs

# %%
df = vandc.collate_runs(runs)[["k", "d", "acc", "max_steps"]].groupby(["k", "d", "max_steps"]).mean().reset_index()

df.to_csv("../results/matched_1048576.csv", index=False)


# %%
import pandas as pd
def read_n(n):
    return pd.read_csv(f"../results/matched_{n}.csv").assign(n = n)

pd.concat([read_n(2 ** k) for k in [8, 12, 16, 20]]).to_csv("../results/matched.csv")
