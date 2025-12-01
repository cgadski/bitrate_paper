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

            bit_ticks = np.arange(1, 7)
            if method_idx == 0:
                ax.set_yticks(bit_ticks / log(2))
                ax.set_yticklabels(bit_ticks)
                ax.set_ylabel(
                    "$d / \\tilde H_2$",
                    rotation="vertical",
                )

            for tick in bit_ticks / log(2):
                ax.axhline(y=tick, color='white', linewidth=0.5, alpha=0.3, zorder=10)

            eta_ticks = [0.1, 0.2, 0.3, 0.4]
            if n_idx == 3:
                ax.set_xticks(eta_ticks)
                ax.set_xlabel(
                    "$\\eta$",
                    rotation="horizontal",
                )

            for tick in eta_ticks:
                ax.axvline(x=tick, color='white', linewidth=0.5, alpha=0.3, zorder=10)

            if n_idx == 0:
                ax.set_title(
                    [
                        "MAP threshold",
                        "Top-$k$",
                        "Matching-$k$\n($2$ steps)",
                        "Matching-$k$\n($3$ steps)",
                        "Matching-$k$\n($64$ steps)",
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
# import vandc
# from pathlib import Path

# def average_results(df):
#     cells = (
#         df[["n", "eta", "factor", "threshold", "max_steps", "k", "d"]]
#         .drop_duplicates()
#         .sort_values(["n", "eta", "factor", "threshold", "max_steps"])
#     )  # pyright: ignore

#     # %%
#     values = (
#         df.groupby(["n", "k", "d", "threshold", "max_steps"])
#         .mean("acc")
#         .reset_index()[["n", "k", "d", "threshold", "max_steps", "acc"]]
#     )
#     values = values.set_index(["n", "k", "d", "threshold", "max_steps"])
#     data = cells.join(values, on=["n", "k", "d", "threshold", "max_steps"], validate="m:1")
#     return data
# setup()
# runs = list(vandc.fetch_dir(Path("../../results/eta_sweep_2")))[:100]
# df = average_results(vandc.collate_runs(vandc.fetch_dir(Path("../../results/eta_sweep_2"))))
# CapacityGraph(df).plot()


# %%
if __name__ == "__main__":
    setup()
    df = pd.concat([
        pd.read_csv("results/eta_sweep_2.csv"),
    ])
    CapacityGraph(df).plot()
    plt.savefig("./figures/eta_sweep.pdf", dpi=300)
