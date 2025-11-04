# %%
from project.graphs.settings import setup, FIG_WIDTH
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def entropy(k, n):
    return k * np.log(np.e * n / k)


def eta(k, n):
    return np.log(k) / np.log(n)


class CapacityGraph:
    def __init__(self, pursuit):
        self.pursuit = pursuit

    def plot_theory(self, ax):
        eta = np.linspace(0, 0.999, 500)
        c = (2 + 4 * np.sqrt(eta) + 2 * eta) / (1 - eta)
        ax.set_ylim(0, 12)
        ax.set_xlim(0, 0.5)

        ax.plot(eta, c)

        ax.set_xlabel("$\\eta$")
        ax.set_ylabel("dims. per nat")

        ax2 = ax.twinx()
        ax2.set_ylim(0, 12 * np.log(2))
        ax2.set_ylabel("dims. per bit")
        ax2.set_yticks(range(0, int(12 * np.log(2)) + 1, 2))

    def plot_data(self, ax, max_steps, label):
        df = self.pursuit
        good = df[
            (df["acc"] > (256 - 8) / 256)
            & (df["n"] == 2**16)
            & (df["max_steps"] == max_steps)
            & (df["k"] > 1)
        ]

        good = good.assign(eta=eta(good["k"], good["n"]))
        good = good.assign(per_nat=good["d"] / entropy(good["k"], good["n"]))
        agg = good.groupby("eta").min()
        ax.scatter(agg.index, agg["per_nat"], s=4, label=label, zorder=10)

    def plot(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(FIG_WIDTH, FIG_WIDTH * 0.8)

        self.plot_theory(ax)

        self.plot_data(ax, 1, "top-$k$")
        self.plot_data(ax, 2, "matching-$k$, 2 steps")
        self.plot_data(ax, 4, "matching-$k$, 4 steps")
        self.plot_data(ax, 64, "matching pursuit")

        ax.legend()
        ax.grid(True)
        fig.tight_layout()


# pursuit = pd.read_csv("../../results/pursuit.csv")
# setup()
# CapacityGraph(pursuit).plot()


# %%
if __name__ == "__main__":
    setup()
    pursuit = pd.read_csv("results/pursuit.csv")
    CapacityGraph(pursuit).plot()
    plt.savefig("figures/capacity.pdf", dpi=300)
