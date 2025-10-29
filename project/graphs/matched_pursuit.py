# %%
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

        # for c in [1, 2, 3, 4]:
        #     p(lambda k, n: c * k * np.log(np.e * n / k), main=False)
        p(lambda k, n: 4 * k * np.log(n * k), main=True)
        # p(lambda k, n: k * np.log(np.e * n / k), main=True)

        ax.set_ylim(0, 1024)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_yticks(2 ** np.arange(8, 11))
        # ax.set_xticks(2 ** np.arange(2, 7))
        # ax.set_ylabel("d")
        # ax.set_xlabel("k")
        ax.set(frame_on=False)
        # ax.set_title("$N = 2^{" + str(int(log2(n))) + "}$")

    def plot(self):
        fig, axs = plt.subplots(4, 4)

        n_vals = [2**8, 2**12, 2**16, 2**20]
        step_vals = [1, 2, 4, 64]

        for arg in grid(n_idx=range(4), step_idx=range(4)):
            n_idx, step_idx = arg["n_idx"], arg["step_idx"]
            n, step = n_vals[n_idx], step_vals[step_idx]
            ax = axs[n_idx][step_idx]

            self.make_subplot(ax, n, step)
            ax.yaxis.tick_right()
            if step_idx == 0:
                ax.set_yticks(2 ** np.arange(7, 11))

            if step_idx == 3:
                ax.yaxis.tick_right()
                ax.set_ylabel(
                    "$N = 2^{" + str(int(log2(n))) + "}$",
                    rotation="horizontal",
                    # fontsize=10,
                    # labelpad=16
                )

            if n_idx == 0:
                ax.set_xticks(2 ** np.arange(3, 7))

        fig.set_size_inches(FIG_WIDTH * 2, FIG_WIDTH * 2)
        fig.tight_layout()
        # plt.tight_layout(rect=[0, 0, 0.9, 1])  # pyright: ignore
        # cbar_ax = fig.add_axes([0.86, 0.17, 0.03, 0.7])  # pyright: ignore
        # fig.colorbar(self.mesh, cax=cbar_ax, label="Success rate")


setup()
c = MatchedPursuit(pd.read_csv("../../results/matched.csv"))
c.plot()

# %%
if __name__ == "__main__":
    setup()
    MatchedPursuit().plot()
    plt.show()
    # plt.savefig("../capacity_paper/figures/matched_pursuit_horiz.pdf", dpi=300)
