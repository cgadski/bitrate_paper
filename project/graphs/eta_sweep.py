# %%
from math import log, log2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.transforms import Affine2D

from project.graphs.settings import C_HUE, FIG_WIDTH, setup
from project.misc import grid


class LogErrorNorm(Normalize):
    def __init__(self, error_floor=1e-4):
        self._log_floor = np.log(error_floor)
        super().__init__(vmin=0, vmax=1)

    def __call__(self, value, clip=None):
        v = np.ma.asarray(value, dtype=float)
        return np.ma.masked_array(
            np.clip(
                np.log(np.clip(1 - v, np.exp(self._log_floor), 1)) / self._log_floor,
                0,
                1,
            )
        )

    def inverse(self, value):
        return 1 - np.exp(np.asarray(value, dtype=float) * self._log_floor)


class CapacityGraph:
    def __init__(self, df):
        self.df = df
        self.hue_norm = LogErrorNorm(error_floor=1e-2)
        self.cmap = LinearSegmentedColormap.from_list("error", ["tab:red", "white"])
        self.cmap.set_bad(color="#e0e0e0")

    def make_subplot(self, ax, n, method):
        df = self.df[(self.df["method"] == method) & (self.df["n"] == n)]

        matrix = df.pivot(index="d_per_nat", columns="eta", values="acc")

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

        eta = np.linspace(0, 0.4, num=128)

        def p(f, main=False):
            opts = {
                "linestyle": (0, (1, 3)),
                "color": "black",
                "lw": 1,
            }
            if main:
                opts["linestyle"] = "--"
            ax.plot(eta, f(eta, n), **opts)

        if method in ["map", "top_k"]:
            p(
                lambda eta, n: (2 + 4 * np.sqrt(eta) + 2 * eta) / (1 - eta),
                main=True,
            )
            p(
                lambda eta, n: 2 / (1 - eta),
                main=False,
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
        fig.set_size_inches(FIG_WIDTH * 2.2, FIG_WIDTH * 1.65)

        n_vals = [2**8, 2**12, 2**16, 2**20]
        method_vals = ["map", "top_k", "2_step", "3_step", "64_step"]

        for arg in grid(n_idx=range(len(n_vals)), method_idx=range(len(method_vals))):
            method_idx: int = arg["method_idx"]  # pyright: ignore
            n_idx: int = arg["n_idx"]  # pyright: ignore
            n, method = n_vals[n_idx], method_vals[method_idx]
            ax = axs[f"{n_idx}_{method_idx}"]

            self.make_subplot(ax, n, method)

            ax.text(
                0.4 * 0.05,
                8.7 * 0.95,
                "N = $2^{" + str(int(log2(n))) + "}$",
                color="black",
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
                ax.axhline(y=tick, color="black", linewidth=0.5, alpha=0.3, zorder=10)

            eta_ticks = [0.1, 0.2, 0.3, 0.4]
            if n_idx == 3:
                ax.set_xticks(eta_ticks)
                ax.set_xlabel(
                    "$\\eta$",
                    rotation="horizontal",
                )

            for tick in eta_ticks:
                ax.axvline(x=tick, color="black", linewidth=0.5, alpha=0.3, zorder=10)

            if n_idx == 0:
                ax.set_title(
                    [
                        "MAP decoding",
                        "Top-$k$",
                        "GMP ($2$ steps)",
                        "GMP ($3$ steps)",
                        "GMP ($64$ steps)",
                    ][method_idx]
                )

        cbar = fig.colorbar(
            self.mesh,
            cax=axs["edge"],
            label="Success rate",
        )
        cbar.set_ticks([0, 0.5, 0.9, 0.99])
        fig.tight_layout()


# %%
# import vandc
# from pathlib import Path


# setup()
# df = pd.read_csv("../../results/eta_sweep.csv")
# CapacityGraph(df).plot()


# %%
if __name__ == "__main__":
    setup()
    df = pd.read_csv("results/eta_sweep_2.csv")
    df = df[df["k"] > 1]

    CapacityGraph(df[df["dict_type"] == "spherical"]).plot()
    plt.savefig("./figures/eta_sweep_spherical.pdf", dpi=300)

    CapacityGraph(df[df["dict_type"] == "rademacher"]).plot()
    plt.savefig("./figures/eta_sweep_rademacher.pdf", dpi=300)
