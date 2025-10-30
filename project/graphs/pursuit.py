# %%
from matplotlib.transforms import Affine2D
import pandas as pd
from matplotlib.colors import Normalize
from project.graphs.settings import setup, FIG_WIDTH, C_HUE
import seaborn as sns
import numpy as np
from math import log2
import matplotlib.pyplot as plt

from project.misc import grid


class MatchedPursuit:
    def __init__(self, df):
        self.df = df
        self.hue_norm = Normalize(0, 1)

    def make_subplot(self, ax, n, max_steps):
        df = self.df[(self.df["n"] == n) & (self.df["max_steps"] == max_steps)]
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

        p(lambda k, n: 1.5 * k * np.log(np.e * n / k), main=True)

        ax.set_ylim(0, 1024)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set(frame_on=False)

    def make_mosaic(self):
        res = []
        for i in range(4):
            last_col = "edge"
            if i in [0, 3]:
                last_col = "."
            res.append([f"{i}_{j}" for j in range(4)] + [last_col])
        return res

    def plot(self):
        fig, axs = plt.subplot_mosaic(
            self.make_mosaic(), width_ratios=[1, 1, 1, 1, 0.1]
        )

        n_vals = [2**8, 2**12, 2**16, 2**20]
        step_vals = [1, 2, 4, 64]

        for arg in grid(n_idx=range(4), step_idx=range(4)):
            step_idx: int = arg["step_idx"]
            n_idx: int = arg["n_idx"]
            n, step = n_vals[n_idx], step_vals[step_idx]
            ax = axs[f"{n_idx}_{step_idx}"]

            self.make_subplot(ax, n, step)

            ax.text(
                64 * 0.95,
                1024 * 0.05,
                "N = $2^{" + str(int(log2(n))) + "}$",
                color="white",
                ha="right",
                va="bottom",
                fontweight="bold",
            )

            if step_idx == 0:
                ax.set_yticks(2 ** np.arange(7, 11))
                ax.set_ylabel(
                    "$d$",
                    rotation="horizontal",
                )

            if n_idx == 3:
                ax.set_xticks(2 ** np.arange(3, 7))
                ax.set_xlabel(
                    "$k$",
                    rotation="horizontal",
                )

            if n_idx == 0:
                ax.set_title(
                    [
                        "top-$k$",
                        "matching-$k$, $2$ steps",
                        "matching-$k$, $4$ steps",
                        "matching pursuit",
                    ][step_idx]
                )

        fig.set_size_inches(FIG_WIDTH * 2, FIG_WIDTH * 1.85)
        axs["edge"]
        fig.colorbar(
            self.mesh,
            cax=axs["edge"],
            label="Success rate",
        )
        fig.tight_layout()


if __name__ == "__main__":
    setup()
    MatchedPursuit(pd.read_csv("results/matched.csv")).plot()
    plt.savefig("./figures/pursuit.pdf", dpi=300)
