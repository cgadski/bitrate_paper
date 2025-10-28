# %%
import pandas as pd
from matplotlib.colors import Normalize
from project.graphs.settings import setup, FIG_WIDTH, C_HUE
import seaborn as sns
import numpy as np
from math import log2
import matplotlib.pyplot as plt


class RustOneStep:
    def __init__(self):
        self.df_low = pd.read_csv("../../results/one_step_low.csv")
        self.df_high = pd.read_csv("../../results/one_step_high.csv")
        self.hue_norm = Normalize(0, 1)

    def make_subplot(self, ax, n, t):
        use_high = n == 1 << 20
        if use_high:
            df = self.df_high
        else:
            df = self.df_low

        df = df[(df["n"] == n) & (df["type"] == t)]
        ax.set_box_aspect(1)

        matrix = df.pivot(index="d", columns="k", values="success_rate")

        self.mesh = ax.pcolormesh(
            matrix.columns,  # k values
            matrix.index,  # d values
            matrix,
            cmap=sns.diverging_palette(C_HUE, 20, as_cmap=True),
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

        if t == "topk":

            def new_pred(k, n):
                eta = np.log(k) / np.log(n)
                return (2 + 4 * np.sqrt(eta) + 2 * eta) * k * np.log(n)

            p(lambda k, n: 4 * k * np.log(k * (n - k)), main=False)
            p(new_pred, main=True)
            # p(lambda k, n: 4 * k * np.log(n))
        else:
            p(lambda k, n: 8 * k * np.log(n), main=True)

        ax.set_yticks(2 ** np.arange(10, 14))
        ax.set_ylim(0, 8200)
        ax.set_ylabel("d")
        ax.set_xlabel("k")
        ax.set(frame_on=False)

    def plot(self):
        n_values = [1 << 8, 1 << 12, 1 << 16, 1 << 20]
        fig, axes = plt.subplots(2, 4)

        for n_index in range(len(n_values)):
            n = n_values[n_index]
            self.make_subplot(axes[0][n_index], n, "threshold")
            self.make_subplot(axes[1][n_index], n, "topk")
            axes[0][n_index].set_title("threshold, $N = 2^{" + str(int(log2(n))) + "}$")
            axes[1][n_index].set_title("top-$k$, $N = 2^{" + str(int(log2(n))) + "}$")

        fig.set_size_inches(FIG_WIDTH * 2.3, FIG_WIDTH)

        plt.tight_layout(rect=[0, 0, 0.9, 1])  # pyright: ignore
        cbar_ax = fig.add_axes([0.91, 0.17, 0.02, 0.7])  # pyright: ignore
        fig.colorbar(self.mesh, cax=cbar_ax, label="Success rate")


# %%
setup()
RustOneStep().plot()


# %%
if __name__ == "__main__":
    setup()
    RustOneStep().plot()
    plt.show()
    # plt.savefig("../capacity_paper/figures/one_step_decode_rs_horiz.pdf", dpi=300)
