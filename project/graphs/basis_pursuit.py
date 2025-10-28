import pandas as pd
from matplotlib.colors import Normalize
from .settings import setup, FIG_WIDTH
import seaborn as sns
import numpy as np
from math import log2
import matplotlib.pyplot as plt


class BpNew:
    def __init__(self):
        self.df = pd.read_csv("results/lasso_2.csv")
        self.mat = (
            self.df.groupby(["k", "d"])["errs"]
            .apply(lambda x: (x == 0).mean())
            .unstack()
        )
        self.hue_norm = Normalize(0, 1)

    def plot(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(FIG_WIDTH, FIG_WIDTH)
        ax.set_box_aspect(1)

        mesh = ax.pcolormesh(
            self.mat.index,  # k values
            self.mat.columns,  # d values
            self.mat.T,
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            norm=self.hue_norm,
            shading="nearest",
            rasterized=True,
        )

        k = np.arange(1, 100)
        ax.plot(
            k,
            k * (1 + np.log((2**16) / k)) / np.log(2),
            linestyle=(0, (1, 3)),
            color="black",
            lw=1,
            alpha=0.5,
        )
        ax.plot(
            k,
            0.8 * k * (1 + np.log((2**16) / k)) / np.log(2),
            "--",
            color="black",
            lw=1,
        )

        ax.set_ylim(0, 1024)
        ax.set_yticks(2 ** np.arange(8, 11))
        ax.set_ylabel("d")
        ax.set_xlabel("k")
        ax.set(frame_on=False)

        fig.colorbar(mesh, fraction=0.04, label="Success rate", orientation="vertical")
        fig.tight_layout()


if __name__ == "__main__":
    setup()
    BpNew().plot()
    plt.show()
    # plt.savefig("../capacity_paper/figures/bp_decode_new.pdf", dpi=300)
