# %%
import pandas as pd
from matplotlib.colors import Normalize
from project.graphs.settings import setup, FIG_WIDTH
import seaborn as sns
import numpy as np
from math import log2
import matplotlib.pyplot as plt
from scipy.special import binom


class StorylineGraph:
    def __init__(self, df):
        self.df = df[df["n"] == 2 ** 20]
        self.df = self.df.drop_duplicates(["k", "d", "max_steps", "threshold"])

    def plot_data(self, ax):
        cutoff = 0.96

        df = self.df[~self.df["threshold"] & (self.df["max_steps"] == 1)]
        df = df[df["acc"] > cutoff].groupby("k").min("d").reset_index()
        ax.scatter(df["k"], df["d"], marker='+', linewidths=1, s=15)

        df = self.df[self.df["max_steps"] == 3]
        df = df[df["acc"] > cutoff].groupby("k").min("d").reset_index()
        ax.scatter(df["k"], df["d"], marker='+', linewidths=1, s=15)

    def plot(self):
        fig, ax = plt.subplots()

        self.plot_data(ax)

        fig.set_size_inches(FIG_WIDTH, FIG_WIDTH * 0.8)
        k = np.linspace(1, 110, 200)
        n = 1 << 20
        ax.set_ylim(0, 2**13)
        ax.set_xlim(0, 100)
        ax.set_yticks(2 ** np.arange(9, 14))

        # Add gridlines
        for tick in 2 ** np.arange(9, 14):
            ax.axhline(y=tick, linestyle='--', linewidth=0.5, color='grey')
        for tick in [20, 40, 60, 80, 100]:
            ax.axvline(x=tick, linestyle='--', linewidth=0.5, color='grey')

        ax2 = ax.twinx()
        ax2.set_ylim(0, 2**13)  # same y-range as left axis
        ax2.set_ylabel("Dimensions per bit")

        one_bit = 100 * (np.log(n / 100) + 1) / np.log(2)
        ax2.set_yticks(np.arange(1, 6) * one_bit)
        ax2.set_yticklabels(np.arange(1, 6))

        eta = np.log(k) / np.log(n)
        ax.plot(k, (2 + 4 * np.sqrt(eta) + 2 * eta) * k * np.log(n), alpha=0.5)
        k_label = 65
        ax.annotate(
            "Top-$k$",
            xy=(k_label, 4 * k_label * np.log(n * k_label)),
            xytext=(-15, 4),
            textcoords="offset points",
        )

        c = 1.75
        ax.plot(k, c * k * (1 + np.log(n / k)) / np.log(2), alpha=0.5)
        k_label = 80
        ax.annotate(
            "gMP ($3$ steps)",
            xy=(k_label, c * k_label * (1 + np.log(n / k_label)) / np.log(2)),
            xytext=(-40, 8),
            textcoords="offset points",
        )

        ax.set_xlabel("$k$")
        ax.set_ylabel("$d$")

        for b in range(1, 7):
            ax.plot(
                k, k * b * (np.log(n / k) + 1) / np.log(2), "--", lw=0.5, color="grey"
            )

        fig.tight_layout()


# %%
# import vandc
# from vandc.writer import git_root
# df = pd.read_csv(git_root() / "results" / "eta_sweep.csv")
# setup()
# StorylineGraph(df).plot()


# %%
if __name__ == "__main__":
    setup()
    StorylineGraph(pd.read_csv("results/eta_sweep.csv")).plot()
    plt.savefig("figures/storyline.pdf", dpi=300)
