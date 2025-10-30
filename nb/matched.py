# %%
%load_ext autoreload
%autoreload 2

# %%
import vandc
import matplotlib.pyplot as plt
import torch as t

# %%
run = runs[1]
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
plt.plot(k, k * t.log2(t.e * 1000000 / k))
plt.plot(k, 4 * k * t.log(1000000 * k))
plt.ylim(0, 1024)

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
