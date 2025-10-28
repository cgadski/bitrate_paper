import pandas as pd
from matplotlib.colors import Normalize
from .settings import setup, FIG_WIDTH, C_HUE
import seaborn as sns
import numpy as np
from math import log2
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from vandc.fetch import read_run
from pathlib import Path

from project.symbol_transmission import Options


class SymbolPlot:
    def __init__(self, n, df: pd.DataFrame, opts: Options):
        self.n = n
        self.df = df
        self.opts = opts
        self.hue_norm = Normalize(0, 1)

    def plot_line(self, ax: Axes, f, bold=False):
        n = 2 ** ((np.arange(0, self.opts.resolution) + 1) * self.opts.n_step)

        opts = {
            "linestyle": (0, (1, 3)),
            "color": "black",
            "lw": 1,
            "alpha": 0.5,
        }
        if bold:
            opts["linestyle"] = "--"
            opts["alpha"] = 1
        ax.plot(n, f(n), **opts)

    def make_subplot(self, ax: Axes, value: str):
        df = self.df
        ax.set_box_aspect(1)

        matrix = df.pivot(index="d", columns="n", values=value)

        self.mesh = ax.pcolormesh(
            matrix.columns,
            matrix.index,
            matrix,
            cmap=sns.diverging_palette(C_HUE, 20, as_cmap=True),
            norm=self.hue_norm,
            shading="nearest",
            rasterized=True,
        )

        ax.set_yticks(2 ** np.arange(4, 9))
        ax.set_ylim(0, self.opts.max_d())
        ax.set_xlim(2, self.opts.max_n())
        ax.set_xscale('log', base=2)
        ax.set_ylabel("$d$")
        ax.set_xlabel("$N$")
        ax.set(frame_on=False)

    def plot(self):
        fig, ax = plt.subplots()

        fig.set_size_inches(FIG_WIDTH * 1.2, FIG_WIDTH)

        self.make_subplot(ax, "success")

        # self.plot_line(axes[0], lambda noise: 8 * noise * np.log(self.n), bold=True)
        self.plot_line(ax, lambda n: 2 * self.opts.noise * np.log(n), bold=True)
        # self.plot_line(
        #     ax,
        #     lambda noise: (0.9 * np.log(self.n) - np.log(2))
        #     / (0.5 * np.log(1 + 1 / noise)),
        # )

        plt.colorbar(self.mesh, ax=ax, label="Success rate")

        plt.tight_layout()


if __name__ == "__main__":
    run = read_run(Path("results/symbol_transmission.csv"))
    print(f"Loading {run}")
    setup()
    SymbolPlot(2 ** 16, run.logs, Options(**run.config)).plot()
    plt.savefig("figures/symbol_transmission.pdf", dpi=300)
