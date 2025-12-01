# %%
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from project.graphs.settings import setup, FIG_WIDTH, C_HUE
import seaborn as sns
import numpy as np
from math import log2, log
import matplotlib.pyplot as plt

from project.misc import grid


class KSweep:
    def __init__(self, df):
        self.df = df
        # self.hue_norm = Normalize(0, 1)
        self.hue_norm = TwoSlopeNorm(0.9, 0, 1)

    def make_subplot(self, ax, n, method):
        df = self.df[(self.df["n"] == n) & (self.df["method"] == method)]
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

        k = np.arange(1, 65)

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
        p(lambda k, n: 4 * k * np.log(k * n))

        ax.set_ylim(0, 2**12)
        ax.set_yticks(2 ** np.arange(10, 13))
        ax.set_xticks(2 ** np.arange(4, 7))
        ax.set_xlabel(
            "$k$",
            rotation="horizontal",
        )
        ax.set(frame_on=False)

        # Add gridlines
        for tick in 2 ** np.arange(10, 12):
            ax.axhline(y=tick, color='white', linewidth=0.5, alpha=0.3, zorder=10)
        for tick in 2 ** np.arange(4, 6):
            ax.axvline(x=tick, color='white', linewidth=0.5, alpha=0.3, zorder=10)

    def plot(self):
        mosaic = [
            ["0_0", "0_1", "0_2", "0_3", "cbar"],
            ["1_0", "1_1", "1_2", "1_3", "cbar"]
        ]

        fig, axs = plt.subplot_mosaic(
            mosaic, # pyright: ignore # pyright: ignore
            width_ratios=[1, 1, 1, 1, 0.1],
            height_ratios=[1, 1]
        ) # pyright: ignore

        n_vals = [2**8, 2**12, 2**16, 2**20]
        methods = ["threshold", "top_k"]

        for row_idx, method in enumerate(methods):
            for col_idx, n in enumerate(n_vals):
                ax = axs[f"{row_idx}_{col_idx}"]
                self.make_subplot(ax, n, method)

                if col_idx > 0:
                    ax.set_yticks([])
                    ax.set_ylabel("")

                if row_idx == 0:
                    ax.set_xlabel("")
                    ax.set_xticks([])

                if row_idx == 0 and col_idx == 0:
                    ax.set_title("MAP threshold", loc="left")
                elif row_idx == 1 and col_idx == 0:
                    ax.set_title("Top-$k$", loc="left")

                ax.text(
                    64 * 0.95,
                    4096 * 0.05,
                    "N = $2^{" + str(int(log2(n))) + "}$",
                    color="white",
                    ha="right",
                    va="bottom",
                    fontweight="bold",
                )

        fig.set_size_inches(FIG_WIDTH * 2, FIG_WIDTH * 1.3 * 0.8)
        cbar = fig.colorbar(
            self.mesh,
            cax=axs["cbar"],
            label="Success rate",
        )
        cbar.set_ticks([0, 0.3, 0.6, 0.9, 0.95, 1])
        fig.tight_layout()


# %%
# import vandc

# setup()
# df = pd.read_csv("../../results/k_sweep.csv")
# # fig, ax = plt.subplots()
# KSweep(df).plot()


# %%
if __name__ == "__main__":
    setup()
    KSweep(pd.read_csv("results/k_sweep.csv")).plot()
    plt.savefig("./figures/k_sweep.pdf", dpi=300)
