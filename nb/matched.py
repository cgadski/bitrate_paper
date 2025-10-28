# %%
%load_ext autoreload
%autoreload 2

# %%
import vandc
import matplotlib.pyplot as plt
import torch as t
from vandc.fetch import read_run
# run = vandc.fetch("leave-natural-way-service") # full matching pursuit
# run = vandc.fetch("send-wrong-single-money") # 4 steps
# run = vandc.fetch("consider-should-late-friend") # 1 step
# run = vandc.fetch("turn-different-place-water") # 2 steps
run = runs[3]
print(run)

matrix = run.logs.pivot(index="d", columns="k", values="acc")

plt.pcolormesh(
    matrix.columns,  # k values
    matrix.index,  # d values
    matrix,
    shading="nearest",
    rasterized=True,
)

k = t.linspace(0, 64, 100)
plt.plot(k, k * t.log2(t.e * 256 / k))

# %%
runs = [run for run in vandc.fetch_all("%matched_pursuit%") if run.config["device"] == 'cuda']
runs

# %%
df = vandc.collate_runs(runs)[["k", "d", "acc", "max_steps"]].groupby(["k", "d", "max_steps"]).mean().reset_index()

df.to_csv("../results/matched_256.csv", index=False)


# %%
df = df[df["max_steps"] == 64]

matrix = df.pivot(index="d", columns="k", values="acc")

plt.pcolormesh(
    matrix.columns,  # k values
    matrix.index,  # d values
    matrix,
    cmap=sns.diverging_palette(220, 20, as_cmap=True),
    shading="nearest",
    rasterized=True,
)

k = t.linspace(0, 64, 100)
plt.plot(k, k * t.log2(t.e * 4096 / k))

# %%
vandc.fetch_all("%matched_pursuit%", this_commit=True)
