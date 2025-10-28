import pandas as pd
from matplotlib.colors import Normalize
from .settings import setup, FIG_WIDTH, C_HUE
import seaborn as sns
import numpy as np
from math import log2
import matplotlib.pyplot as plt


class MatchedPursuit:
    def __init__(self):
        self.df = pd.read_csv("results/mp.csv")
        self.df_large = pd.read_csv("results/mp_large.csv")
        self.hue_norm = Normalize(0, 1)

    def make_subplot(self, ax, n):
        use_large = n == (1 << 20)

        if use_large:
            df = self.df_large
            df = df[df["k"] % 2 == 0]
        else:
            df = self.df[self.df["n"] == n]
        df = df.groupby(["d", "k"])["accuracy"].mean().reset_index()
        ax.set_box_aspect(1)

        matrix = df.pivot(index="d", columns="k", values="accuracy")

        if use_large:
            k_vals = matrix.columns
            d_vals = matrix.index
            theoretical = lambda k: k * (1 + np.log(n / k)) / np.log(2)

            for k in k_vals:
                for d in d_vals:
                    if np.isnan(matrix.loc[d, k]):
                        matrix.loc[d, k] = 1.0 if d >= theoretical(k) else 0.0

        self.mesh = ax.pcolormesh(
            matrix.columns,  # k values
            matrix.index,  # d values
            matrix,
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            norm=self.hue_norm,
            shading="nearest",
            rasterized=True,
        )

        k = np.arange(1, 100)

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

        # p(lambda k, n: 4 * k * np.log(n))
        # p(lambda k, n: np.log(binom(n, k))/np.log(2), main=True)
        p(lambda k, n: 1.3 * k * (1 + np.log(n / k)) / np.log(2), main=False)
        p(lambda k, n: k * (1 + np.log(n / k)) / np.log(2), main=True)
        # p(lambda k, n: 8 * k * np.log(n))

        ax.set_ylim(0, 2048)
        ax.set_yticks(2 ** np.arange(8, 12))
        ax.set_ylabel("d")
        ax.set_xlabel("k")
        ax.set(frame_on=False)
        ax.set_title("$N = 2^{" + str(int(log2(n))) + "}$")

    def plot(self):
        fig, ax = plt.subplots(2, 2)

        self.make_subplot(ax[0][0], 1 << 8)
        self.make_subplot(ax[0][1], 1 << 12)
        self.make_subplot(ax[1][0], 1 << 16)
        self.make_subplot(ax[1][1], 1 << 20)

        fig.set_size_inches(FIG_WIDTH * 1.3, FIG_WIDTH)
        fig.tight_layout()
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # pyright: ignore
        cbar_ax = fig.add_axes([0.86, 0.17, 0.03, 0.7])  # pyright: ignore
        fig.colorbar(self.mesh, cax=cbar_ax, label="Success rate")


if __name__ == "__main__":
    setup()
    MatchedPursuit().plot()
    plt.show()
    # plt.savefig("../capacity_paper/figures/matched_pursuit_horiz.pdf", dpi=300)
