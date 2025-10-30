# %%
import pandas as pd
from matplotlib.colors import Normalize
from project.graphs.settings import setup, FIG_WIDTH, C_HUE
import seaborn as sns
import numpy as np
from math import log2
import matplotlib.pyplot as plt

from project.misc import grid


class TopK:
    def __init__(self, df):
        self.df = df
        self.hue_norm = Normalize(0, 1)

    def make_subplot(self, ax, n):
        df = self.df[self.df["n"] == n]
        matrix = df.pivot(index="d", columns="k", values="acc")

        ax.set_box_aspect(1)
        self.mesh = ax.pcolormesh(
            matrix.columns,
            matrix.index,
            matrix,
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            norm=self.hue_norm,
            shading="nearest",
            rasterized=True,
        )

        k = np.arange(1, 64)

        def p(f, main=False):
            opts = {
                "linestyle": (0, (1, 3)),
                "color": "black",
                "lw": 1,
                "alpha": 0.5,
            }
            if main:
                opts["linestyle"] = "--"
                opts["alpha"] = 1
            ax.plot(k, f(k, n), **opts)

        def upper(k, n):
            eta = np.log(k) / np.log(n)
            c = 2 + 4 * np.sqrt(eta) + 2 * eta
            return c * k * np.log(n)

        p(upper, main=True)

        for c in [6]:
            p(lambda k, n: c * k * np.log(np.e * n / k))

        ax.set_ylim(0, 2**12)
        ax.set_yticks(2 ** np.arange(8, 13))
        ax.set_xticks(2 ** np.arange(3, 7))
        ax.set_xlabel(
            "$k$",
            rotation="horizontal",
        )
        ax.set(frame_on=False)

    def plot(self):
        fig, axs = plt.subplots(1, 5, width_ratios=[1, 1, 1, 1, 0.1])

        n_vals = [2**12, 2**14, 2**16, 2**18]

        for i, n in enumerate(n_vals):
            ax = axs[i]
            self.make_subplot(ax, n)

            if i > 0:
                ax.set_yticks([])

            ax.text(
                64 * 0.95,
                4096 * 0.05,
                "N = $2^{" + str(int(log2(n))) + "}$",
                color="white",
                ha="right",
                va="bottom",
                fontweight="bold",
            )

        fig.set_size_inches(FIG_WIDTH * 2, FIG_WIDTH * 0.7 * 0.8)
        fig.colorbar(
            self.mesh,
            cax=axs[-1],
            label="Success rate",
        )
        fig.tight_layout()


# %%
if __name__ == "__main__":
    setup()
    TopK(pd.read_csv("results/top_k.csv")).plot()
    plt.savefig("./figures/top_k.pdf", dpi=300)
