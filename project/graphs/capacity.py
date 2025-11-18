# %%
from project.graphs.settings import setup, FIG_WIDTH, C_HUE
import seaborn as sns
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def entropy(k, n):
    return k * (1 + np.log(n) - np.log(k))


def eta(k, n):
    return np.log(k) / np.log(n)


class CapacityGraph:
    def __init__(self, sweep):
        self.sweep = sweep
        self.hue_norm = TwoSlopeNorm(0.9, 0, 1)

    def plot_theory(self, ax):
        eta = np.linspace(0, 0.999, 500)
        c = (2 + 4 * np.sqrt(eta) + 2 * eta) / (1 - eta)
        # c = (4 + 4 * eta) / (1 - eta)
        ax.set_ylim(0, 9)
        ax.set_xlim(0, 0.4)

        ax.plot(eta, c)

        ax.set_xlabel("$\\eta$")
        ax.set_ylabel("dims. per nat")

        # ax2 = ax.twinx()
        # ax2.set_ylim(0, 12 * np.log(2))
        # ax2.set_ylabel("dims. per bit")
        # ax2.set_yticks(range(0, int(12 * np.log(2)) + 1, 2))

    def plot_data(self, ax):
        matrix = self.sweep.pivot(index="factor", columns="eta", values="acc")
        self.mesh = ax.pcolormesh(
            matrix.columns,  # eta
            matrix.index,  # factor
            matrix,
            cmap=sns.diverging_palette(C_HUE, 20, as_cmap=True),
            norm=self.hue_norm,
            rasterized=True,
        )

    def plot(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(FIG_WIDTH, FIG_WIDTH * 0.8)

        self.plot_theory(ax)
        self.plot_data(ax)

        # ax.legend()
        ax.grid(True)
        fig.tight_layout()


# %%
import vandc
from project.pursuit import threshold, rademacher, DTYPE
import torch as t
from pathlib import Path

run = list(vandc.fetch_dir(Path("../../results/eta_sweep/")))[100]
print(run.config)
# pursuit = vandc.fetch("offer-foreign-result-question").logs
# pursuit = vandc.fetch("play-only-year-member").logs
# # pursuit = vandc.fetch("remember-strong-business-government").logs
# # pursuit = vandc.fetch().logs
# # pursuit = vandc.fetch("start-popular-woman-line").logs
CapacityGraph(run.logs).plot()
# %%
from math import exp, log

k = 16
4 * k * log(1024)

# %%
import vandc

run = vandc.fetch()
plt.matshow(run.logs.pivot(index="factor", columns="eta", values="acc"))


# %%
if __name__ == "__main__":
    setup()
    pursuit = pd.read_csv("results/pursuit.csv")
    CapacityGraph(pursuit).plot()
    plt.savefig("figures/capacity.pdf", dpi=300)
