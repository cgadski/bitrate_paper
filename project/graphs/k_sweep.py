# %%
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from project.graphs.eta_sweep import LogErrorNorm
from project.graphs.settings import setup, FIG_WIDTH, C_HUE
import seaborn as sns
import numpy as np
from math import log2, log
import matplotlib.pyplot as plt

from project.misc import grid


class KSweep:
    def __init__(self, df):
        self.df = df
        self.hue_norm = LogErrorNorm(error_floor=1e-2)
        self.cmap = LinearSegmentedColormap.from_list("error", ["tab:red", "white"])
        self.cmap.set_bad(color="#e0e0e0")

    def make_subplot(self, ax, n, method):
        df = self.df[(self.df["n"] == n) & (self.df["method"] == method)]
        matrix = df.pivot(index="d", columns="k", values="acc")

        ax.set_box_aspect(1)
        self.mesh = ax.pcolormesh(
            matrix.columns,
            matrix.index,
            matrix,
            cmap=self.cmap,
            norm=self.hue_norm,
            shading="nearest",
            rasterized=True,
        )

        k = np.linspace(1, 65)

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

        # eta = np.log(64) / np.log(n)
        # c = 2 + 4 * np.sqrt(eta) + 2 * eta
        p(upper, main=True)
        # p(lambda k, n: c * k * np.log(n))
        p(lambda k, n: 8 * k * np.log(n))
        p(lambda k, n: 4 * k * np.log(n))

        ax.set_ylim(0, 2**12)
        ax.set_yticks(2 ** np.arange(10, 13))
        ax.set_xticks(2 ** np.arange(4, 7))
        ax.set_xlabel(
            "$k$",
            rotation="horizontal",
        )
        ax.set_ylabel("$d$")
        ax.set(frame_on=False)

        # Add gridlines
        for tick in 2 ** np.arange(10, 12):
            ax.axhline(
                y=tick, color="black", linewidth=0.5, alpha=0.3, zorder=10, clip_on=True
            ).set_clip_path(ax.patch)
        for tick in 2 ** np.arange(4, 6):
            ax.axvline(
                x=tick, color="black", linewidth=0.5, alpha=0.3, zorder=10, clip_on=True
            ).set_clip_path(ax.patch)

    def plot(self):
        mosaic = [
            ["0_0", "0_1", "0_2", "0_3", "cbar"],
            ["1_0", "1_1", "1_2", "1_3", "cbar"],
        ]

        fig, axs = plt.subplot_mosaic(
            mosaic,  # pyright: ignore # pyright: ignore
            width_ratios=[1, 1, 1, 1, 0.1],
            height_ratios=[1, 1],
        )  # pyright: ignore

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
                    ax.set_title("MAP decoding", loc="left")
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

        fig.set_size_inches(FIG_WIDTH * 1.8, FIG_WIDTH * 0.9)
        cbar = fig.colorbar(
            self.mesh,
            cax=axs["cbar"],
            label="Success rate",
        )
        cbar.set_ticks([0, 0.5, 0.9, 0.99])
        fig.tight_layout()


# import vandc

# setup()
# df = pd.read_csv("../../results/k_sweep.csv")
# KSweep(df).plot()


# %%
if __name__ == "__main__":
    setup()
    KSweep(pd.read_csv("results/k_sweep.csv")).plot()
    plt.savefig("./figures/k_sweep.pdf", dpi=300)
