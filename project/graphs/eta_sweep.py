# %%
from matplotlib.transforms import Affine2D
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from project.graphs.settings import setup, FIG_WIDTH, C_HUE
import seaborn as sns
import numpy as np
from math import log2, log
import matplotlib.pyplot as plt

from project.misc import grid


class CapacityGraph:
    def __init__(self, df):
        self.df = df
        # self.hue_norm = Normalize(0, 1)
        self.hue_norm = TwoSlopeNorm(0.9, 0, 1)

    def make_subplot(self, ax, n, method):
        if method == "threshold":
            df = self.df[(self.df["n"] == n) & self.df["threshold"]]
        else:
            df = self.df[
                (self.df["n"] == n)
                & (self.df["max_steps"] == method)
                & ~self.df["threshold"]
            ]

        matrix = df.pivot(index="factor", columns="eta", values="acc")

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

        eta = np.linspace(0, 0.4, num=128)

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
            ax.plot(eta, f(eta, n), **opts)

        p(
            lambda eta, n: (2 + 4 * np.sqrt(eta) + 2 * eta) / (1 - eta + (1 / log(n))),
            main=True,
        )
        p(
            lambda eta, n: (2 + 4 * np.sqrt(eta) + 2 * eta) / (1 - eta)
        )
        p(lambda eta, n: (2 / log(2)) * np.ones_like(eta))

        ax.set_ylim(0, 9)
        ax.set_xlim(0, 0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set(frame_on=False)

    def make_mosaic(self):
        res = []
        for i in range(4):
            last_col = "edge"
            if i in [0, 3]:
                last_col = "."
            res.append([f"{i}_{j}" for j in range(5)] + [last_col])
        return res

    def plot(self):
        fig, axs = plt.subplot_mosaic(
            self.make_mosaic(), width_ratios=[1, 1, 1, 1, 1, 0.1]
        )
        fig.set_size_inches(FIG_WIDTH * 2.4, FIG_WIDTH * 1.85)

        n_vals = [2**8, 2**12, 2**16, 2**20]
        method_vals = ["threshold", 1, 2, 3, 64]

        for arg in grid(n_idx=range(len(n_vals)), method_idx=range(len(method_vals))):
            method_idx: int = arg["method_idx"]  # pyright: ignore
            n_idx: int = arg["n_idx"]  # pyright: ignore
            n, method = n_vals[n_idx], method_vals[method_idx]
            ax = axs[f"{n_idx}_{method_idx}"]

            self.make_subplot(ax, n, method)

            ax.text(
                0.4 * 0.05,
                9 * 0.95,
                "N = $2^{" + str(int(log2(n))) + "}$",
                color="white",
                ha="left",
                va="top",
                fontweight="bold",
            )

            if method_idx == 0:
                ax.set_yticks([1, 3, 5, 7, 9])
                ax.set_ylabel(
                    "$d / \\tilde H$",
                    rotation="vertical",
                )

            if n_idx == 3:
                ax.set_xticks([0.1, 0.2, 0.3, 0.4])
                ax.set_xlabel(
                    "$\\eta$",
                    rotation="horizontal",
                )

            if n_idx == 0:
                ax.set_title(
                    [
                        "MAP threshold",
                        "top-$k$",
                        "matching-$k$\n($2$ steps)",
                        "matching-$k$\n($3$ steps)",
                        "matching-$k$\n($64$ steps)",
                    ][method_idx]
                )

        cbar = fig.colorbar(
            self.mesh,
            cax=axs["edge"],
            label="Success rate",
        )
        cbar.set_ticks([0, 0.3, 0.6, 0.9, 0.95, 1])
        fig.tight_layout()


# %%
# setup()
# CapacityGraph(df).plot()


# %%
if __name__ == "__main__":
    setup()
    df = pd.concat([
        pd.read_csv("results/eta_sweep.csv"),
        pd.read_csv("results/eta_sweep_mid.csv"),
    ])
    CapacityGraph(df).plot()
    plt.savefig("./figures/eta_sweep.pdf", dpi=300)
