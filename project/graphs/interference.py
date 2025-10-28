import pandas as pd
from matplotlib.colors import Normalize
from .settings import setup, FIG_WIDTH
import seaborn as sns
import numpy as np
from math import log2
import matplotlib.pyplot as plt


class GdInterference:
    def __init__(self, df: pd.DataFrame):
        self.df = df.loc[df["final_loss"] < 1 / 2]

    def plot_separate(self, ax):
        sns.lineplot(
            self.df, x=self.df["d"], y=self.df["final_loss"] ** 2, hue="n", ax=ax
        )
        ax.legend(title="$N$")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        # ax.set_aspect(1)
        ax.set_ylim(2**-12, None)
        ax.grid(True)
        d_range = 2 ** np.linspace(4, 10, num=20)
        ax.plot(d_range, 1 / d_range, ":", color="grey")
        ax.set_ylabel("Optimal $\\gamma$")
        ax.set_xlabel("Embedding dimension (d)")

    def plot_rescaled(self, ax):
        sns.lineplot(
            self.df,
            x=self.df["d"] / self.df["n"],
            y=(self.df["final_loss"] ** 2) * self.df["d"],
            ax=ax,
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_ylim(2**-4, None)
        ax.set_xlim(2**-6, None)
        ax.grid(True)
        ax.set_ylabel("Ratio $\\gamma_{\\mathrm{opt}} / \\gamma_{\\mathrm{init}}$")
        ax.set_xlabel("Ratio $d/N$")

    def plot(self):
        fig, ax = plt.subplots(2, 1)
        fig.set_size_inches(FIG_WIDTH, FIG_WIDTH * 1.2)
        self.plot_separate(ax[0])
        self.plot_rescaled(ax[1])
        fig.tight_layout()


if __name__ == "__main__":
    setup()
    GdInterference(pd.read_csv("results/gd_interference.csv")).plot()
    plt.show()
    # plt.savefig("../capacity_paper/figures/gd_interference.pdf", dpi=300)
